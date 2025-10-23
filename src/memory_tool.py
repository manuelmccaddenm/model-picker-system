from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_memory(memory_path: str | Path) -> Dict[str, Any]:
    p = Path(memory_path)
    if not p.exists():
        return {"schema_version": "1.0", "updated_at": _now_iso(), "teoria": [], "experiencias": [], "lecciones_analista": []}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(memory_path: str | Path, data: Dict[str, Any]) -> None:
    p = Path(memory_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data.setdefault("schema_version", "1.0")
    data["updated_at"] = _now_iso()
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_schema(memory_path: str | Path) -> Dict[str, Any]:
    mem = load_memory(memory_path)
    changed = False
    if "schema_version" not in mem:
        mem["schema_version"] = "1.0"
        changed = True
    if "updated_at" not in mem:
        mem["updated_at"] = _now_iso()
        changed = True
    for key, default in (
        ("teoria", []),
        ("experiencias", []),
        ("lecciones_analista", []),
    ):
        if key not in mem or not isinstance(mem[key], list):
            mem[key] = default
            changed = True
    if changed:
        save_memory(memory_path, mem)
    return mem


def build_indexes(mem: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    by_task: Dict[str, List[str]] = {}
    by_tag: Dict[str, List[str]] = {}
    for m in mem.get("teoria", []):
        mid = m.get("model_id")
        task = m.get("task")
        if mid and task:
            by_task.setdefault(task, []).append(mid)
        for tag in m.get("tags", []) or []:
            by_tag.setdefault(tag, []).append(mid)
    return {"by_task": by_task, "by_tag": by_tag}


def get_theory(mem: Dict[str, Any], task: Optional[str] = None, tags: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items = mem.get("teoria", [])
    if task:
        items = [m for m in items if m.get("task") == task]
    if tags:
        tagset = set(tags)
        items = [m for m in items if tagset.intersection(set(m.get("tags", [])))]
    if limit is not None:
        items = items[:limit]
    return items


def rank_candidates_by_J(models: List[Dict[str, Any]], J_weights: Dict[str, float]) -> List[Tuple[str, float, Dict[str, Any]]]:
    ranked: List[Tuple[str, float, Dict[str, Any]]] = []
    for m in models:
        scores = m.get("tradeoff_scores", {}) or {}
        total = 0.0
        for k, w in J_weights.items():
            try:
                total += float(scores.get(k, 0.0)) * float(w)
            except Exception:
                continue
        ranked.append((m.get("model_id", "unknown"), float(total), m))
    ranked.sort(key=lambda t: t[1], reverse=True)
    return ranked


def get_experiences(mem: Dict[str, Any], task: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
    exps = mem.get("experiencias", [])
    if task:
        exps = [e for e in exps if e.get("task") == task]
    return exps[:k]


def add_experience(memory_path: str | Path, experience: Dict[str, Any]) -> str:
    mem = ensure_schema(memory_path)
    exp = dict(experience)
    exp.setdefault("id", f"exp_{uuid.uuid4().hex}")
    exp.setdefault("timestamp", _now_iso())
    mem.setdefault("experiencias", []).append(exp)
    save_memory(memory_path, mem)
    return exp["id"]


def add_lesson(memory_path: str | Path, lesson: Dict[str, Any]) -> str:
    mem = ensure_schema(memory_path)
    ls = dict(lesson)
    ls.setdefault("id", f"lesson_{uuid.uuid4().hex}")
    ls.setdefault("last_updated", _now_iso())
    mem.setdefault("lecciones_analista", []).append(ls)
    save_memory(memory_path, mem)
    return ls["id"]
