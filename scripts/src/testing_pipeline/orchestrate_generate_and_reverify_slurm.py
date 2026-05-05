#!/usr/bin/env python3
"""
Submit Slurm generation, wait, then a one-task-at-a-time verification array.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]


def _solutions_dir_from_config(repo_root: Path, config_path: Path, experiment_id: str) -> Path:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset_path = Path(cfg["dataset_path"])
    split = cfg["split"]
    dataset_name = cfg.get("dataset_name") or dataset_path.stem
    return repo_root / "results" / str(dataset_name) / str(split) / str(experiment_id)


def _problem_dirs_with_attempts(solutions_dir: Path, attempts_file: str = "attempts.jsonl") -> list[Path]:
    if not solutions_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(solutions_dir.iterdir()):
        if p.is_dir() and (p / attempts_file).is_file():
            out.append(p)
    return out


def _submit_sbatch(
    repo_root: Path,
    script: Path,
    *,
    export_pairs: dict[str, str],
    array_spec: str | None = None,
    output_pattern: str | None = None,
    error_pattern: str | None = None,
) -> str:
    exports = ["ALL"]
    for k, v in export_pairs.items():
        exports.append(f"{k}={v}")
    cmd = ["sbatch", "--parsable", f"--export={','.join(exports)}"]
    if array_spec is not None:
        cmd.append(f"--array={array_spec}")
    if output_pattern is not None:
        cmd.append(f"--output={output_pattern}")
    if error_pattern is not None:
        cmd.append(f"--error={error_pattern}")
    cmd.append(str(script))
    r = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise SystemExit(f"sbatch failed ({r.returncode}): {r.stderr or r.stdout}")
    job_id = (r.stdout or "").strip()
    if not re.fullmatch(r"[0-9]+", job_id):
        raise SystemExit(f"Unexpected sbatch output: {r.stdout!r}")
    return job_id


def _wait_slurm_job_gone(job_id: str, poll_seconds: float) -> None:
    while True:
        r = subprocess.run(["squeue", "-j", job_id, "-h"], capture_output=True, text=True, check=False)
        if not (r.stdout or "").strip():
            time.sleep(2.0)
            r2 = subprocess.run(["squeue", "-j", job_id, "-h"], capture_output=True, text=True, check=False)
            if not (r2.stdout or "").strip():
                return
        time.sleep(poll_seconds)


_BAD_SACCT_STATES = frozenset({"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"})


def _any_slurm_non_completed(job_id: str) -> tuple[bool, str]:
    r = subprocess.run(
        ["sacct", "-j", job_id, "-n", "-P", "-o", "JobID,State,ExitCode"],
        capture_output=True,
        text=True,
        check=False,
    )
    bad_bits: list[str] = []
    for ln in (r.stdout or "").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("JobID"):
            continue
        parts = ln.split("|")
        if len(parts) < 3:
            continue
        jid, state, exitcode = parts[0], parts[1], parts[2]
        if "extern" in jid:
            continue
        if state in _BAD_SACCT_STATES:
            bad_bits.append(f"{jid}:{state}:{exitcode}")
        elif state == "COMPLETED" and exitcode and exitcode != "0:0":
            ec_main = exitcode.split(":", 1)[0]
            if ec_main != "0":
                bad_bits.append(f"{jid}:{state}:{exitcode}")
    if bad_bits:
        return True, " ; ".join(bad_bits[:20])
    return False, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="generate_solutions config YAML")
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--sleep-after-gen", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument(
        "--gen-script",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "bash" / "testing_pipeline" / "slurm" / "generate_solutions_gpu.sh",
    )
    parser.add_argument(
        "--verify-script",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "bash" / "testing_pipeline" / "slurm" / "verify_one_problem_array_adhoc.sh",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    cfg_path = args.config if args.config.is_absolute() else (repo_root / args.config).resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    experiment_id = args.experiment_id or str(uuid.uuid4())
    solutions_dir = _solutions_dir_from_config(repo_root, cfg_path, experiment_id)
    log_dir = repo_root / ".slurm" / "pipeline" / experiment_id
    log_dir.mkdir(parents=True, exist_ok=True)

    gen_script = args.gen_script if args.gen_script.is_absolute() else (repo_root / args.gen_script).resolve()
    verify_script = args.verify_script if args.verify_script.is_absolute() else (repo_root / args.verify_script).resolve()

    rel_config = cfg_path
    try:
        rel_config = cfg_path.relative_to(repo_root)
    except ValueError:
        pass

    if not args.skip_generation:
        gen_job = _submit_sbatch(
            repo_root,
            gen_script,
            export_pairs={"REPO_ROOT": str(repo_root), "GEN_CONFIG": str(rel_config), "GEN_EXPERIMENT_ID": str(experiment_id)},
            output_pattern=str(log_dir / "%x_%j.out"),
            error_pattern=str(log_dir / "%x_%j.err"),
        )
        _wait_slurm_job_gone(gen_job, args.poll_seconds)
        bad, msg = _any_slurm_non_completed(gen_job)
        if bad:
            raise SystemExit(f"Generation job {gen_job} did not complete successfully: {msg}")
        time.sleep(max(0.0, float(args.sleep_after_gen)))

    n = len(_problem_dirs_with_attempts(solutions_dir))
    if n == 0:
        raise SystemExit(f"No problem subdirs with attempts.jsonl under {solutions_dir}")

    verify_job = _submit_sbatch(
        repo_root,
        verify_script,
        export_pairs={"REPO_ROOT": str(repo_root), "SOLUTIONS_DIR": str(solutions_dir)},
        array_spec=f"0-{n - 1}%1",
        output_pattern=str(log_dir / "%x_%A_%a.out"),
        error_pattern=str(log_dir / "%x_%A_%a.err"),
    )
    _wait_slurm_job_gone(verify_job, args.poll_seconds)
    bad, msg = _any_slurm_non_completed(verify_job)
    if bad:
        raise SystemExit(f"Verification job {verify_job} had non-COMPLETED tasks: {msg}")

    if not args.no_merge:
        merge = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "src" / "testing_pipeline" / "merge_reverification_by_problem.py"), str(solutions_dir)],
            cwd=repo_root,
            check=False,
        )
        if merge.returncode != 0:
            raise SystemExit(f"merge_reverification_by_problem.py failed ({merge.returncode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
