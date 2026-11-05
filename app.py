"""Local web interface for the folder agent.

  .venv/bin/streamlit run app.py --server.address localhost
"""
import time
from pathlib import Path

import ollama
import streamlit as st

import agent
from folder_search import connect, search

TOOL_LABELS = {
    "search_documents": "Searching",
    "read_file": "Reading",
    "list_files": "Listing files",
    "grep": "Finding exact text",
}

st.set_page_config(page_title="Folder Search", layout="centered")
st.title("Folder Search")

db = connect()
files = db.execute("SELECT COUNT(*) FROM files WHERE status = 'indexed'").fetchone()[0]
st.caption(f"{files} documents indexed. Everything stays on this Mac.")

mode = st.radio(
    "Mode",
    ["Ask the agent", "Quick search"],
    horizontal=True,
    captions=["Written answer, takes several minutes", "Matching excerpts, takes seconds"],
)

with st.form("ask"):
    question = st.text_input("Your question")
    submitted = st.form_submit_button("Go", type="primary")

if submitted and question.strip():
    if mode == "Quick search":
        with st.spinner("Searching..."):
            results = search(question)
        if not results:
            st.info("No matches found.")
        for r in results:
            with st.container(border=True):
                st.markdown(f"**{Path(r['path']).name}** · {r['location']}")
                st.text(r["text"][:400])
                st.caption(r["path"])
    else:
        started = time.time()
        with st.status("Working...", expanded=True) as status:
            def show_tool(name, args):
                detail = args.get("query") or args.get("text") or args.get("pattern") or ""
                if args.get("path"):
                    detail = Path(args["path"]).name
                st.write(f"{TOOL_LABELS.get(name, name)}: {detail}")

            try:
                answer = agent.run(question, on_tool=show_tool)
            except ConnectionError:
                status.update(label="Could not reach Ollama", state="error")
                st.error("Make sure the Ollama app is open, then try again.")
                st.stop()
            except ollama.ResponseError as e:
                status.update(label="Ollama error", state="error")
                st.error(e.error)
                st.stop()
            status.update(label=f"Done in {time.time() - started:.0f} seconds", state="complete")
        st.subheader("Answer")
        st.markdown(answer)
        st.caption("Check important details against the cited files. Small local models can make mistakes.")
