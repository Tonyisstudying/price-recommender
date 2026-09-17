from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil
import tempfile
from typing import Any


class ModelRegistry:
    """Simple local model registry with atomic promotion.

    Models are trained into immutable version directories. The `current.json`
    manifest is replaced atomically only after a challenger passes evaluation.
    The API can therefore keep serving the previous champion if an update fails.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.versions = self.root / "versions"
        self.current_file = self.root / "current.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.versions.mkdir(parents=True, exist_ok=True)

    def new_version_id(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def version_path(self, version_id: str) -> Path:
        path = self.versions / version_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_current(self) -> dict[str, Any] | None:
        if not self.current_file.exists():
            return None
        return json.loads(self.current_file.read_text(encoding="utf-8"))

    def promote(self, version_id: str, metadata: dict[str, Any]) -> None:
        payload = {
            "version": version_id,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        fd, tmp = tempfile.mkstemp(
            prefix="current_",
            suffix=".json",
            dir=self.root,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.current_file)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def load_current_metadata(self) -> tuple[str, dict[str, Any]]:
        current = self.get_current()
        if not current:
            raise FileNotFoundError("No promoted model version exists.")
        version = current["version"]
        return version, current

    def cleanup_old_versions(self, keep: int = 5) -> None:
        dirs = sorted(
            [p for p in self.versions.iterdir() if p.is_dir()],
            key=lambda p: p.name,
            reverse=True,
        )
        for path in dirs[keep:]:
            shutil.rmtree(path, ignore_errors=True)

