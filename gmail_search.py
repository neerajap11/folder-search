#!/usr/bin/env python3
"""Read-only Gmail tools, using Google's official API.

  python gmail_search.py login
  python gmail_search.py search "from:someone@example.com newer_than:30d"

The only permission requested is gmail.readonly, so these tools cannot send,
change or delete email. Credentials stay in ~/.folder-search, outside the repo.
"""
import argparse
import base64
import json
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CONFIG_DIR = Path.home() / ".folder-search"
CLIENT_FILE = CONFIG_DIR / "google_client.json"
TOKEN_FILE = CONFIG_DIR / "gmail_token.json"
MAX_RESULTS = 10
READ_CHARS = 4000


def credentials():
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        if not CLIENT_FILE.exists():
            raise FileNotFoundError(f"Google client file not found at {CLIENT_FILE}")
        creds = InstalledAppFlow.from_client_secrets_file(str(CLIENT_FILE), SCOPES).run_local_server(port=0)
    TOKEN_FILE.touch(mode=0o600, exist_ok=True)
    TOKEN_FILE.write_text(creds.to_json())
    os.chmod(TOKEN_FILE, 0o600)
    return creds


def service():
    return build("gmail", "v1", credentials=credentials(), cache_discovery=False)


def header(message, name):
    for h in message["payload"].get("headers", []):
        if h["name"].lower() == name:
            return h["value"]
    return ""


def plain_text(part):
    if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(part["body"]["data"]).decode(errors="ignore")
    return "\n".join(filter(None, (plain_text(p) for p in part.get("parts", []))))


def search_gmail(query):
    """Returns matching emails, newest first. Accepts Gmail search syntax."""
    gmail = service()
    found = gmail.users().messages().list(userId="me", q=query, maxResults=MAX_RESULTS).execute()
    results = []
    for item in found.get("messages", []):
        m = gmail.users().messages().get(
            userId="me", id=item["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()
        results.append({
            "id": m["id"],
            "from": header(m, "from"),
            "subject": header(m, "subject"),
            "date": header(m, "date"),
            "snippet": m.get("snippet", ""),
        })
    return json.dumps(results, ensure_ascii=False)


def read_email(email_id, offset=0):
    """Returns an email's plain text, READ_CHARS at a time."""
    m = service().users().messages().get(userId="me", id=email_id, format="full").execute()
    text = plain_text(m["payload"])
    offset = max(0, int(offset))
    end = offset + READ_CHARS
    return json.dumps({
        "id": email_id,
        "from": header(m, "from"),
        "subject": header(m, "subject"),
        "date": header(m, "date"),
        "text": text[offset:end],
        "next_offset": end if end < len(text) else None,
    }, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("login")
    sub.add_parser("search").add_argument("query")
    args = parser.parse_args()
    if args.command == "login":
        credentials()
        print("Connected to Gmail with read-only access.")
    else:
        for r in json.loads(search_gmail(args.query)):
            print(f"{r['date']} | {r['from']} | {r['subject']}")
