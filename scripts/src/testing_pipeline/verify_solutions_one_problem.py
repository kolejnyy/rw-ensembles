#!/usr/bin/env python3
"""Verify one problem's attempts under a solutions directory (single process, one verifier)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from invpro.utils.metrics import pass_at_k
from invpro.utils.verifier import ProofVerifier


def _parse_k_values(csv: str) -> list[int]:
    out: list[int] = []
    for p in [x.strip() for x in csv.split(",") if x.strip()]:
        k = int(p)
        if k <= 0:
            raise SystemExit(f"k must be positive; got {k}")
        if k not in out:
            out.append(k)
    return out


def _is_lsp_crash_like_error(err: Optional[str]) -> bool:
    if not err:
        return False
    s = str(err).lower()
    return (
        "-32901" in s
        or "-32902" in s
        or ("server process" in s and "crashed" in s)
        or "lsp error" in s
        or "broken pipe" in s
        or "connection reset" in s
    )


def _verify_one(
    verifier: ProofVerifier,
    project_root: str,
    initial_imports: str,
    timeout_seconds: float,
    idx: int,
    code: str,
) -> tuple[bool, str, ProofVerifier]:
    """
    Verify one attempt; reset verifier on LSP crash. Returns (ok, err_message, verifier).
    """
    try:
        ok, err = verifier.verify(code)
    except Exception as e:
        err_s = str(e)
        if _is_lsp_crash_like_error(err_s):
            verifier.close()
            verifier = ProofVerifier(
                project_root=project_root,
                initial_imports=initial_imports,
                timeout_seconds=timeout_seconds,
            )
        return False, err_s, verifier
    if (not ok) and _is_lsp_crash_like_error(err):
        verifier.close()
        verifier = ProofVerifier(
            project_root=project_root,
            initial_imports=initial_imports,
            timeout_seconds=timeout_seconds,
        )
        ok, err = verifier.verify(code)
    return ok, "" if err is None else str(err), verifier


def _read_attempts(attempts_path: Path, max_attempts: Optional[int]) -> list[str]:
    attempts: list[tuple[int, str]] = []
    with attempts_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = rec.get("final_code") or ""
            if not isinstance(code, str) or not code.strip():
                continue
            an = rec.get("attempt_num")
            try:
                idx = int(an) if an is not None else len(attempts)
            except Exception:
                idx = len(attempts)
            attempts.append((idx, code))
            if max_attempts is not None and len(attempts) >= int(max_attempts):
                break
    attempts.sort(key=lambda x: x[0])
    return [c for _, c in attempts]


def _problem_dirs_with_attempts(solutions_dir: Path, attempts_file: str) -> list[Path]:
    return [
        p
        for p in sorted(solutions_dir.iterdir())
        if p.is_dir() and (p / attempts_file).is_file()
    ]


def _problem_dirs_from_list_file(list_file: Path) -> list[Path]:
    out: list[Path] = []
    with list_file.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(Path(s))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solutions_dir", type=Path)
    parser.add_argument("--problem-index", type=int, required=True)
    parser.add_argument("--attempts-file", type=str, default="attempts.jsonl")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--preamble", type=str, default="import Mathlib\n")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--k-values", type=str, default="1,2,4,8,16,32,64,128")
    parser.add_argument("--max-attempts-per-problem", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--problem-list-file",
        type=Path,
        default=None,
        help=(
            "Optional text file with one absolute problem dir path per line. "
            "If set, problem-index is resolved against this list (for resumable/chunked arrays)."
        ),
    )
    args = parser.parse_args()

    solutions_dir = args.solutions_dir.resolve()
    if args.problem_list_file is not None:
        problem_dirs = _problem_dirs_from_list_file(args.problem_list_file.resolve())
    else:
        problem_dirs = _problem_dirs_with_attempts(solutions_dir, args.attempts_file)
    problem_dir = problem_dirs[args.problem_index]
    problem_name = problem_dir.name
    codes = _read_attempts(problem_dir / args.attempts_file, args.max_attempts_per_problem)
    if not codes:
        raise SystemExit("No non-empty attempts")

    project_root = str(args.project_root)
    preamble = args.preamble
    timeout = float(args.timeout_seconds)

    verifier = ProofVerifier(
        project_root=project_root,
        initial_imports=preamble,
        timeout_seconds=timeout,
    )
    successes: list[Optional[bool]] = [None] * len(codes)
    failures: list[tuple[int, str]] = []
    try:
        for idx, code in enumerate(codes):
            ok, err, verifier = _verify_one(
                verifier, project_root, preamble, timeout, idx, code
            )
            successes[idx] = ok
            if not ok:
                failures.append((idx + 1, err))
    finally:
        verifier.close()

    success_bools = [bool(x) for x in successes]
    success_ids = [i + 1 for i, ok in enumerate(success_bools) if ok]
    k_values = _parse_k_values(args.k_values)
    per_k = {f"pass@{k}": pass_at_k(success_bools, k) for k in k_values}
    out_dir = args.output_dir or (solutions_dir / ".reverification_by_problem")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{problem_name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "problem": problem_name,
                "problem_index": args.problem_index,
                "n_attempts": len(success_bools),
                "verified_success_count": len(success_ids),
                "successful_attempt_ids_1based": success_ids,
                "successes": success_bools,
                "pass_at_k": per_k,
                "failures": [{"attempt_1based": i, "error": err[:2000]} for i, err in failures],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")
    print(f"Successful attempt ids (1-based): {success_ids}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
