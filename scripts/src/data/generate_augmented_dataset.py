"""
Generate augmented dataset from full.lean files.

Manager script: discovers files, shuffles by seed, skips already-processed,
then runs worker subprocesses in parallel (ThreadPoolExecutor). Each worker
processes one file, runs its own Lean LSP client, prints a JSON result, and
exits. The manager enforces per-file timeouts and kills the process group
on hang.

Workers are implemented in invpro.dataset.generate_augmented.

Run with:
  conda run -n invpro python scripts/generate_augmented_dataset.py --file-workers 4 --out-dir .augmented_dataset
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from invpro.dataset.generate_augmented import (
    _infer_uuid_from_path,
    _output_dir_exists,
)

WORKER_MODULE = "invpro.dataset.generate_augmented"
DEFAULT_WORKER_TIMEOUT = 600


def _list_full_lean_files(raw_root: Path) -> List[Path]:
    """Return all `<raw_root>/<uuid>/full.lean` paths that exist."""
    if not raw_root.is_dir():
        return []
    out: List[Path] = []
    for child in raw_root.iterdir():
        if not child.is_dir():
            continue
        p = child / "full.lean"
        if p.is_file():
            out.append(p)
    return out


def _kill_worker(proc: subprocess.Popen) -> None:
    """Kill worker process and its process group (e.g. Lean children)."""
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def run_worker_subprocess(
    lean_path: Path,
    project_root: str,
    out_dir: Path,
    timeout: int = DEFAULT_WORKER_TIMEOUT,
) -> Dict[str, Any]:
    """
    Run a worker subprocess to process one file. Returns a result dict
    (JSON from worker, or {"ok": False, "reason": "timeout"|"crashed"|"bad-json", ...}).
    """
    lean_path = lean_path.resolve()
    out_dir = out_dir.resolve()
    project_path = Path(project_root).resolve()
    uuid = _infer_uuid_from_path(lean_path)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            WORKER_MODULE,
            str(lean_path),
            str(project_path),
            str(out_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=project_root,
        start_new_session=True,
    )

    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_worker(proc)
        try:
            _out, _err = proc.communicate()
            err = _err
        except Exception:
            err = b""
        return {
            "ok": False,
            "uuid": uuid,
            "reason": "timeout",
            "stderr": err.decode(errors="replace") if err else "",
        }
    except (BrokenPipeError, ConnectionResetError):
        _kill_worker(proc)
        try:
            _out, _err = proc.communicate()
            err = _err
        except Exception:
            err = b""
        return {
            "ok": False,
            "uuid": uuid,
            "reason": "broken_pipe",
            "stderr": err.decode(errors="replace") if err else "",
        }

    stderr_text = err.decode(errors="replace")
    stdout_text = out.decode(errors="replace")

    if proc.returncode != 0:
        try:
            data = json.loads(stdout_text)
            return data
        except Exception:
            pass
        return {
            "ok": False,
            "uuid": uuid,
            "reason": "crashed",
            "stderr": stderr_text,
            "stdout": stdout_text,
        }

    try:
        return json.loads(stdout_text)
    except Exception as e:
        return {
            "ok": False,
            "uuid": uuid,
            "reason": "bad-json",
            "stdout": stdout_text,
            "stderr": stderr_text,
            "parse_error": repr(e),
        }


class _FileLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._fh.write(f"[{ts}] {msg}\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


_LOGGER: Optional[_FileLogger] = None


def _log(msg: str) -> None:
    if _LOGGER is not None:
        _LOGGER.log(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data") / "NuminaMath-LEAN" / "raw",
        help="Root directory containing raw UUID folders",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".augmented_dataset"),
        help="Output directory for augmented dataset entries",
    )
    parser.add_argument(
        "--file-workers",
        type=int,
        default=1,
        help="Number of worker subprocesses to run in parallel (default: 1)",
    )
    parser.add_argument(
        "--worker-timeout",
        type=int,
        default=DEFAULT_WORKER_TIMEOUT,
        help=f"Timeout in seconds per file (default: {DEFAULT_WORKER_TIMEOUT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling files (default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of files to process",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional path to progress log file (default: <out-dir>/generate.log)",
    )

    args = parser.parse_args()

    global _LOGGER
    log_path = args.log if args.log is not None else args.out_dir / "generate.log"
    _LOGGER = _FileLogger(log_path)
    _log(f"starting dataset generation (out={args.out_dir}, log={log_path})")

    project_root = str(Path(".").resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"[scan] listing full.lean files from {args.raw_root}")
    files = _list_full_lean_files(args.raw_root)
    if not files:
        raise SystemExit(f"No full.lean files found under: {args.raw_root}")

    rng = random.Random(args.seed)
    rng.shuffle(files)

    if args.limit is not None:
        files = files[: args.limit]

    n_before = len(files)
    files = [p for p in files if not _output_dir_exists(p, args.out_dir)]
    n_skipped = n_before - len(files)
    if n_skipped > 0:
        _log(f"[scan] skipping {n_skipped} file(s) already present in {args.out_dir}")
    if not files:
        _log("[run] no files left to process")
        _log("=== Results ===")
        _log("files processed: 0")
        _log("files with errors: 0")
        _log("total entries saved: 0")
        _log(f"output directory: {args.out_dir}")
        if _LOGGER is not None:
            _LOGGER.close()
            _LOGGER = None
        return 0

    n_files = len(files)
    _log(
        f"[run] processing {n_files} file(s) with {args.file_workers} worker(s), "
        f"timeout={args.worker_timeout}s per file"
    )

    total_saved = 0
    processed = 0
    errors = 0
    done_count = 0

    def _run_one(p: Path) -> Tuple[Path, Dict[str, Any]]:
        res = run_worker_subprocess(
            lean_path=p,
            project_root=project_root,
            out_dir=args.out_dir,
            timeout=args.worker_timeout,
        )
        return (p, res)

    with ThreadPoolExecutor(max_workers=args.file_workers) as executor:
        future_to_path = {executor.submit(_run_one, p): p for p in files}
        for future in as_completed(future_to_path):
            p = future_to_path[future]
            try:
                _, res = future.result()
            except Exception as exc:
                res = {"ok": False, "uuid": _infer_uuid_from_path(p), "error": repr(exc)}

            if res.get("ok"):
                total_saved += res.get("saved", 0)
                processed += 1
            else:
                errors += 1
                reason = res.get("reason", "error")
                errmsg = res.get("error", res.get("parse_error", ""))
                try:
                    _log(f"[error] {p}: {reason}" + (f" — {errmsg}" if errmsg else ""))
                except Exception:
                    pass

            done_count += 1
            if (done_count % 5 == 0) or (done_count == n_files):
                try:
                    _log(
                        f"[progress] {done_count} completed, "
                        f"{n_files - done_count} remaining, {total_saved} entries saved"
                    )
                except Exception:
                    pass

    _log("=== Results ===")
    _log(f"files processed: {processed}")
    _log(f"files with errors: {errors}")
    _log(f"total entries saved: {total_saved}")
    _log(f"output directory: {args.out_dir}")

    if _LOGGER is not None:
        _LOGGER.close()
        _LOGGER = None

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
