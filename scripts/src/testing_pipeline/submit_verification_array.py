#!/usr/bin/env python3
"""Submit a Slurm array for per-problem re-verification on an existing solutions folder."""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]


def _problem_dirs_with_attempts(
    solutions_dir: Path,
    attempts_file: str,
    *,
    skip_existing: bool,
    output_dir: Path | None,
) -> list[Path]:
    out: list[Path] = []
    out_dir = output_dir or (solutions_dir / ".reverification_by_problem")
    for p in sorted(solutions_dir.iterdir()):
        if not (p.is_dir() and (p / attempts_file).is_file()):
            continue
        if skip_existing and (out_dir / f"{p.name}.json").is_file():
            continue
        out.append(p)
    return out


def _submit_sbatch(
    repo_root: Path,
    verify_script: Path,
    *,
    solutions_dir: Path,
    array_spec: str,
    output_pattern: str,
    error_pattern: str,
    rwens_python: str | None,
    problem_index_offset: int,
    dependency_after_job_id: str | None,
    dependency_type: str,
    problem_list_file: Path | None,
) -> str:
    exports = ["ALL", f"REPO_ROOT={repo_root}", f"SOLUTIONS_DIR={solutions_dir}"]
    if problem_index_offset:
        exports.append(f"VERIFY_PROBLEM_INDEX_OFFSET={problem_index_offset}")
    if problem_list_file is not None:
        exports.append(f"VERIFY_PROBLEM_LIST_FILE={problem_list_file}")
    if rwens_python:
        exports.append(f"RWENS_PYTHON={rwens_python}")
    cmd = [
        "sbatch",
        "--parsable",
    ]
    if dependency_after_job_id:
        cmd.append(f"--dependency={dependency_type}:{dependency_after_job_id}")
    cmd.extend(
        [
            f"--array={array_spec}",
            f"--output={output_pattern}",
            f"--error={error_pattern}",
            f"--export={','.join(exports)}",
            str(verify_script),
        ]
    )
    r = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise SystemExit(f"sbatch failed ({r.returncode}): {r.stderr or r.stdout}")
    job_id = (r.stdout or "").strip()
    if not re.fullmatch(r"[0-9]+", job_id):
        raise SystemExit(f"Unexpected sbatch output: {r.stdout!r}")
    return job_id


def _wait_for_slurm_job(job_id: str, *, poll_s: float = 15.0) -> None:
    """Block until ``squeue`` no longer lists this job (completed, failed, or cancelled)."""
    while True:
        r = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%i"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0 or not (r.stdout or "").strip():
            return
        time.sleep(poll_s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solutions_dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--verify-script",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "bash" / "testing_pipeline" / "slurm" / "verify_one_problem_array_adhoc.sh",
    )
    parser.add_argument("--attempts-file", type=str, default="attempts.jsonl")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip problems that already have .reverification_by_problem/<problem>.json. "
            "Default: on (use --no-skip-existing to force re-verification)."
        ),
    )
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument(
        "--max-array-tasks",
        type=int,
        default=1000,
        help=(
            "Max Slurm array task count per submitted job. Cluster MaxArraySize is often ~1001; "
            "larger verification runs are split into multiple array jobs with offsets."
        ),
    )
    parser.add_argument(
        "--parallel-chunks",
        action="store_true",
        help=(
            "Submit all chunk jobs immediately (they may run concurrently). "
            "Default is to chain chunks with Slurm dependencies so chunk 2 starts after chunk 1 finishes."
        ),
    )
    parser.add_argument(
        "--chunk-dependency",
        choices=["afterany", "afterok"],
        default="afterany",
        help=(
            "Dependency type used between chunked array jobs. "
            "'afterany' (default) starts the next chunk when the prior chunk finishes "
            "regardless of success/failure; 'afterok' requires full success."
        ),
    )
    parser.add_argument("--log-tag", type=str, default=None)
    parser.add_argument("--rwens-python", type=str, default=None)
    parser.add_argument(
        "--wait",
        action="store_true",
        help="After submitting, block until each verification Slurm job finishes (poll squeue).",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    solutions_dir = args.solutions_dir if args.solutions_dir.is_absolute() else (repo_root / args.solutions_dir).resolve()
    if not solutions_dir.is_dir():
        raise SystemExit(f"Not a directory: {solutions_dir}")
    verify_script = args.verify_script if args.verify_script.is_absolute() else (repo_root / args.verify_script).resolve()
    if not verify_script.is_file():
        raise SystemExit(f"Verify script not found: {verify_script}")

    all_dirs = _problem_dirs_with_attempts(solutions_dir, args.attempts_file, skip_existing=False, output_dir=None)
    all_n = len(all_dirs)
    todo_dirs = _problem_dirs_with_attempts(
        solutions_dir,
        args.attempts_file,
        skip_existing=bool(args.skip_existing),
        output_dir=None,
    )
    n = len(todo_dirs)
    if n == 0:
        if all_n == 0:
            raise SystemExit(f"No problem subdirs with {args.attempts_file!r} under {solutions_dir}")
        print(
            "Nothing to submit: all problems already have .reverification_by_problem/*.json "
            f"(total={all_n}, skip_existing={bool(args.skip_existing)})."
        )
        return 0
    max_conc = max(1, int(args.max_concurrent))
    chunk = max(1, int(args.max_array_tasks))

    log_tag = args.log_tag or solutions_dir.name
    log_dir = repo_root / ".slurm" / "pipeline" / log_tag
    log_dir.mkdir(parents=True, exist_ok=True)
    # Freeze the exact todo list so chunked arrays index into the same subset.
    task_list_path = log_dir / "verification_problem_list.txt"
    task_list_path.write_text("".join(f"{p.resolve()}\n" for p in todo_dirs), encoding="utf-8")

    job_ids: list[str] = []
    prev_job_id: str | None = None
    for offset in range(0, n, chunk):
        chunk_n = min(chunk, n - offset)
        # Slurm rejects e.g. ``0-0%4`` when only one task: throttle must not exceed array size.
        simultaneous = min(max_conc, chunk_n)
        array_spec = f"0-{chunk_n - 1}%{simultaneous}"
        use_dep = prev_job_id is not None and not args.parallel_chunks
        job_id = _submit_sbatch(
            repo_root,
            verify_script,
            solutions_dir=solutions_dir,
            array_spec=array_spec,
            output_pattern=str(log_dir / "%x_%A_%a.out"),
            error_pattern=str(log_dir / "%x_%A_%a.err"),
            rwens_python=args.rwens_python,
            problem_index_offset=offset,
            dependency_after_job_id=prev_job_id if use_dep else None,
            dependency_type=args.chunk_dependency,
            problem_list_file=task_list_path,
        )
        job_ids.append(job_id)
        dep_note = f" {args.chunk_dependency}:{prev_job_id}" if use_dep else ""
        print(
            f"Submitted verification array job {job_id}{dep_note} — "
            f"problems {offset}..{offset + chunk_n - 1} ({chunk_n} tasks)"
        )
        prev_job_id = job_id
    if len(job_ids) > 1:
        mode = "parallel chunks" if args.parallel_chunks else "sequential chunks (dependency chain)"
        print(
            f"Total array jobs submitted: {len(job_ids)} "
            f"(n_todo={n}, n_total={all_n}, chunk={chunk}, {mode}, skip_existing={bool(args.skip_existing)})"
        )
    else:
        print(f"Submitted 1 array job (n_todo={n}, n_total={all_n}, skip_existing={bool(args.skip_existing)})")
    if args.wait and job_ids:
        for jid in job_ids:
            print(f"Waiting for Slurm job {jid} ...", flush=True)
            _wait_for_slurm_job(jid)
        print("All submitted verification jobs finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
