# Folder Search

Ask questions about a messy folder of documents, using RAG and an agent that runs entirely on your own machine.

![Folder Search interface](docs/screenshot.jpg)

## What it does

- **Quick search:** returns matching excerpts in seconds, combining search by meaning with search by keywords.
- **Ask the agent:** a local model searches, reads files and writes an answer that names its sources. It can take several minutes on modest hardware.

The agent is read-only, and it can only open files that are in the index.

## How it works

1. **Index:** extracts text from PDF, Word, PowerPoint, text and Markdown files, splits it into chunks, and stores keywords and embeddings in a local SQLite file. Re-indexing only processes changed files.
2. **Retrieve:** merges the meaning and keyword rankings with reciprocal rank fusion.
3. **Agent:** a loop of up to 10 steps where the model picks a tool (`search_documents`, `read_file`, `list_files` or `grep`) and decides whether to keep looking.

## Setup

Requires Python 3.12 and [Ollama](https://ollama.com).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
ollama pull qwen3:4b
```

## Usage

```bash
.venv/bin/python folder_search.py index /path/to/your/folder
.venv/bin/streamlit run app.py
```

You can also use the command line:

```bash
.venv/bin/python folder_search.py search "your query"
.venv/bin/python agent.py "your question"
```

## Privacy

Documents and the index stay on your machine. The index, `index.db`, holds a plain-text copy of your documents' text, so never commit or share it. The `.gitignore` here blocks everything except the listed code files.

## Limitations

- Scanned PDFs need OCR, which isn't supported yet.
- Spreadsheets, CSVs and images aren't indexed.
- Small local models are slow on 8 GB of memory, and can stop searching too early, so check answers against the cited files.
- PyMuPDF is AGPL or commercially licensed, so check its terms before hosting or distributing this.
