from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from ..config import OPENAI_MODEL


def _get_client() -> OpenAI:
    return OpenAI()


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start_idx = text.find("```")
    while start_idx != -1:
        end_idx = text.find("```", start_idx + 3)
        if end_idx == -1:
            break
        candidate = text[start_idx + 3 : end_idx]
        if "\n" in candidate:
            first_line, rest = candidate.split("\n", 1)
            if first_line.lower().strip() in {"json", "javascript", "ts", "python"}:
                candidate = rest
        try:
            return json.loads(candidate)
        except Exception:
            pass
        start_idx = text.find("```", end_idx + 3)
    return {}


def upload_file_for_assistants(file_path: str | Path) -> str:
    client = _get_client()
    with Path(file_path).open("rb") as f:
        created = client.files.create(file=f, purpose="assistants")
    return created.id


def create_assistant(name: str, instructions: str, tools: Optional[List[Dict[str, Any]]] = None, model: Optional[str] = None) -> Any:
    client = _get_client()
    assistant = client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model or OPENAI_MODEL,
        tools=tools or [],
    )
    return assistant


def run_assistant_with_messages(
    assistant_id: str,
    messages: List[Dict[str, Any]],
    file_tool_map: Optional[Dict[str, List[str]]] = None,
    poll_interval_seconds: float = 0.8,
    timeout_seconds: float = 120.0,
) -> Dict[str, Any]:
    client = _get_client()
    thread_messages: List[Dict[str, Any]] = []
    for m in messages:
        msg: Dict[str, Any] = {"role": m.get("role", "user"), "content": m["content"]}
        attachments = []
        for file_id, tools in (file_tool_map or {}).items():
            attachments.append({"file_id": file_id, "tools": [{"type": tool} for tool in tools]})
        if attachments:
            msg["attachments"] = attachments
        thread_messages.append(msg)

    thread = client.beta.threads.create(messages=thread_messages)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)

    start = time.time()
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status in {"completed", "failed", "cancelled", "expired"}:
            break
        if time.time() - start > timeout_seconds:
            break
        time.sleep(poll_interval_seconds)

    messages_list = client.beta.threads.messages.list(thread_id=thread.id).data
    for message in messages_list:
        if message.role == "assistant":
            for part in message.content:
                if part.type == "text":
                    text = part.text.value or ""
                    parsed = extract_json_from_text(text)
                    if parsed:
                        return parsed
    return {}


def llm_chat(messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.2) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model or OPENAI_MODEL,
        temperature=temperature,
        messages=messages,
    )
    return response.choices[0].message.content or ""
