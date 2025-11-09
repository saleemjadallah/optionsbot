"""Local favorites fallback store used when the backend service is unreachable."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List


class LocalFavoriteStore:
    """
    Persist favorites to a JSON file so the UI can keep working even when the
    remote favorites service (Railway deployment) is offline.
    """

    def __init__(self, storage_path: str | Path | None = None):
        env_path = os.getenv("FAVORITES_CACHE_FILE")
        if storage_path is not None:
            self.path = Path(storage_path).expanduser()
        elif env_path:
            self.path = Path(env_path).expanduser()
        else:
            self.path = Path.home() / ".options_trader" / "favorites.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def list(self, account_number: str) -> List[Dict[str, Any]]:
        store = self._read_store()
        items = store.get(account_number, [])
        return [dict(item) for item in items]

    def replace_all(self, account_number: str, favorites: List[Dict[str, Any]]) -> None:
        with self._lock:
            store = self._read_store()
            store[account_number] = favorites
            self._write_store(store)

    def save(self, account_number: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        saved = dict(payload)
        saved.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        with self._lock:
            store = self._read_store()
            items = store.setdefault(account_number, [])
            for idx, existing in enumerate(items):
                if existing.get("idea_id") == saved.get("idea_id"):
                    items[idx] = saved
                    break
            else:
                items.append(saved)
            self._write_store(store)
        return saved

    def delete(self, account_number: str, idea_id: str) -> bool:
        with self._lock:
            store = self._read_store()
            items = store.get(account_number, [])
            updated = [item for item in items if item.get("idea_id") != idea_id]
            if len(updated) == len(items):
                return False
            store[account_number] = updated
            self._write_store(store)
        return True

    def _read_store(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, OSError):
            pass
        return {}

    def _write_store(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        tmp_path.replace(self.path)
