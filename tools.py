import json
from typing import List

import pandas as pd
from langchain_core.tools import tool
from psycopg2 import sql
from rapidfuzz import fuzz

from db import get_connection


@tool
def entity_fuzzy_matcher_tool(search_term: str, columns_to_search: List[str]) -> str:
    """Normalize user input (slang, abbreviations, partial names) into exact
    database values by fuzzy-matching against distinct values in public.reports.

    Args:
        search_term: The raw term the user mentioned (e.g. "גולני", "מילואים").
        columns_to_search: Column names to fetch and compare against,
            e.g. ['shem_misgeret', 'shem_chativa', 'shem_ugda'].
    Returns:
        JSON string of the best matching row as {column: value} dict,
        or a message if nothing matched well.
    """
    if not columns_to_search:
        return json.dumps({"error": "columns_to_search must not be empty"})

    first_col = columns_to_search[0]
    cols = sql.SQL(", ").join(sql.Identifier(c) for c in columns_to_search)
    query = sql.SQL(
        "SELECT DISTINCT {cols} FROM public.reports WHERE {first} IS NOT NULL"
    ).format(cols=cols, first=sql.Identifier(first_col))

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return json.dumps({"message": "No data found for the requested columns."})

    search_lower = search_term.lower()
    best_score = 0
    best_row = None

    for row in rows:
        values = [str(v) for v in row if v is not None]
        combined = " ".join(values)
        combined_lower = combined.lower()

        # Exact substring containment gets a big boost
        score = fuzz.WRatio(search_term, combined)
        for val in values:
            val_lower = val.lower().strip()
            if val_lower and (val_lower in search_lower or search_lower in val_lower):
                score = max(score, 95)
                break

        if score > best_score:
            best_score = score
            best_row = dict(zip(columns_to_search, row))

    if best_score < 40:
        return json.dumps(
            {"message": f"No good match found for '{search_term}' (best score {best_score})."}
        )

    return json.dumps(best_row, ensure_ascii=False, default=str)


@tool
def database_sql_executor_tool(query: str) -> str:
    """Execute a read-only PostgreSQL query against public.reports and return
    the results as a JSON array of objects.

    Args:
        query: A fully formed PostgreSQL SELECT query.
    Returns:
        JSON string containing the query results as a list of dicts,
        or an error message.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=columns)
        return df.to_json(orient="records", force_ascii=False, date_format="iso")
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    finally:
        conn.close()
