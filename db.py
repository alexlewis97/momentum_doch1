import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """Return a psycopg2 connection using the Supabase connection string."""
    password = os.getenv("DB_PASSWORD")
    return psycopg2.connect(
        user="postgres",
        password=password,
        host="db.nnctnxwlnkspkdfxnozg.supabase.co",
        port="5432",
        dbname="postgres",
    )
