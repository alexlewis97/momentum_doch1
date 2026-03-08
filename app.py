import json

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent import build_agent, get_system_message

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="IDF HR Insights", layout="centered")

# ── Minimal custom CSS ───────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {max-width: 760px; padding-top: 2rem;}
    .stChatMessage {font-family: 'Inter', sans-serif;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("IDF HR Insights")

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = [get_system_message()]


def render_assistant_content(text: str):
    """Parse assistant text; render __SQL__ blocks as code and __TABLE__ blocks as dataframes."""
    remaining = text
    while remaining:
        # Find the next special tag
        sql_idx = remaining.find("__SQL__")
        tbl_idx = remaining.find("__TABLE__")

        # No more tags — render the rest as markdown
        if sql_idx == -1 and tbl_idx == -1:
            if remaining.strip():
                st.markdown(remaining.strip())
            break

        # Determine which tag comes first
        if sql_idx != -1 and (tbl_idx == -1 or sql_idx < tbl_idx):
            # Render text before the tag
            before = remaining[:sql_idx].strip()
            if before:
                st.markdown(before)
            remaining = remaining[sql_idx + len("__SQL__"):]
            # Extract SQL until next tag or end
            next_tag = min(
                (remaining.find(t) for t in ("__SQL__", "__TABLE__") if remaining.find(t) != -1),
                default=len(remaining),
            )
            sql_text = remaining[:next_tag].strip()
            st.code(sql_text, language="sql")
            remaining = remaining[next_tag:]
        else:
            # Render text before the tag
            before = remaining[:tbl_idx].strip()
            if before:
                st.markdown(before)
            remaining = remaining[tbl_idx + len("__TABLE__"):]
            # Extract JSON until next tag or end
            next_tag = min(
                (remaining.find(t) for t in ("__SQL__", "__TABLE__") if remaining.find(t) != -1),
                default=len(remaining),
            )
            json_str = remaining[:next_tag].strip()
            try:
                data = json.loads(json_str)
                if isinstance(data, list) and data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, hide_index=True, use_container_width=True)
                else:
                    st.markdown(json_str)
            except (json.JSONDecodeError, ValueError):
                st.markdown(json_str)
            remaining = remaining[next_tag:]


# ── Render chat history ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_content(msg["content"])
        else:
            st.markdown(msg["content"])

# ── Chat input ───────────────────────────────────────────────────────────────
if prompt := st.chat_input("שאל שאלה על נתוני כוח אדם..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build LangChain messages
    st.session_state.lc_messages.append(HumanMessage(content=prompt))

    # Invoke agent
    with st.chat_message("assistant"):
        with st.spinner("מעבד..."):
            result = st.session_state.agent.invoke(
                {"messages": st.session_state.lc_messages}
            )
            # Extract the final AI response
            ai_messages = [
                m for m in result["messages"] if isinstance(m, AIMessage)
            ]
            if ai_messages:
                answer = ai_messages[-1].content
            else:
                answer = "לא הצלחתי לעבד את הבקשה."

            render_assistant_content(answer)

    # Persist
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.lc_messages = result["messages"]
