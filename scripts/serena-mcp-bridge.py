#!/usr/bin/env python3
"""
MCP stdio-to-streamable-http bridge for Serena.
Starts instantly and proxies MCP JSON-RPC messages to the Serena HTTP MCP server
running on localhost:8001, handling session management.
"""
import sys
import json
import http.client
import threading

SERENA_HOST = "127.0.0.1"
SERENA_PORT = 8001
SERENA_PATH = "/mcp"

session_id = None
session_lock = threading.Lock()

def parse_sse_response(body):
    """Parse SSE response body and return JSON messages."""
    messages = []
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            try:
                msg = json.loads(line[6:])
                messages.append(msg)
            except json.JSONDecodeError:
                pass
    if not messages:
        # Try parsing as regular JSON
        try:
            msg = json.loads(body)
            messages.append(msg)
        except json.JSONDecodeError:
            pass
    return messages

def send_to_serena(request):
    global session_id
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    with session_lock:
        if session_id:
            headers["Mcp-Session-Id"] = session_id

    body = json.dumps(request)
    conn = http.client.HTTPConnection(SERENA_HOST, SERENA_PORT, timeout=120)
    try:
        conn.request("POST", SERENA_PATH, body=body, headers=headers)
        resp = conn.getresponse()

        # Extract session ID from response
        new_session_id = resp.getheader("Mcp-Session-Id")
        if not new_session_id:
            cookie = resp.getheader("Set-Cookie")
            if cookie:
                for part in cookie.split(";"):
                    part = part.strip()
                    if part.startswith("session_id="):
                        new_session_id = part[len("session_id="):]
                        break

        if new_session_id:
            with session_lock:
                if not session_id:
                    session_id = new_session_id

        response_body = resp.read().decode("utf-8")
        messages = parse_sse_response(response_body)
        return messages
    finally:
        conn.close()

def send_response(response_data):
    sys.stdout.write(json.dumps(response_data) + "\n")
    sys.stdout.flush()

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            send_response({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
            continue

        if request.get("jsonrpc") == "2.0":
            try:
                messages = send_to_serena(request)
                for msg in messages:
                    send_response(msg)
            except Exception as e:
                send_response({"jsonrpc": "2.0", "id": request.get("id"), "error": {"code": -32000, "message": str(e)}})

if __name__ == "__main__":
    main()
