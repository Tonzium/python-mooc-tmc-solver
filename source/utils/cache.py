# utils/cache.py
"""
Cache helpers for JSON artifacts.

For users:
- The app stores scraped data under ./data to speed up subsequent runs.

For developers:
- get_or_build_json(path, builder, ...): returns cached JSON if fresh;
  otherwise calls builder(), writes atomically, and returns the new data.
- 'max_age_days' lets you expire stale caches without manual deletion.

Implementation details:
- Writes are atomic (tmp + move) to avoid partial files on crashes.
- 'exercise_path()' creates a predictable filename per exercise anchor.
"""

from __future__ import annotations
from pathlib import Path
import json, shutil, time
from typing import Any, Callable, Dict, List, Union

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EX_DIR   = DATA_DIR / "exercises"
for p in (DATA_DIR, EX_DIR):
    p.mkdir(parents=True, exist_ok=True)

def exercise_path(exercise_id: str) -> Path:
    safe = exercise_id.replace("/", "_")
    return EX_DIR / f"{safe}.json"

# ---------------------------------------------------------------------
# Write JSON via temp file + rename to ensure readers never see a half-written file.
# ---------------------------------------------------------------------
def _write_json_atomic(data: Union[Dict[str, Any], List[Any]], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    shutil.move(str(tmp), str(path))

def _read_json(path: Path) -> Union[Dict[str, Any], List[Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------
## Expiry helper: treat cache as stale if file age exceeds 'max_age_days'.
# ---------------------------------------------------------------------
def _is_expired(path: Path, max_age_days: int | None) -> bool:
    if max_age_days is None:
        return False
    age = time.time() - path.stat().st_mtime
    return age > max_age_days * 24 * 3600

def get_or_build_json(path: Path, builder: Callable[[], Union[Dict[str, Any], List[Any]]],
                      *, force: bool = False, max_age_days: int | None = None):
    if path.exists() and not force and not _is_expired(path, max_age_days):
        return _read_json(path)
    data = builder()
    _write_json_atomic(data, path)
    return data

def index_path() -> Path:
    # project_root/data/index.json
    from pathlib import Path
    return (Path(__file__).resolve().parents[1] / "data" / "index.json")