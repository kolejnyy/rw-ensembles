"""
Single-file processing for augmented dataset generation.

Processes one full.lean file: split proof into tactics, fetch states via LSP,
run VariableRenamer, write tactic_XXXX.txt entries under out_dir/<uuid>/.

Can be run as a worker subprocess:
  python -m rwens.dataset.generate_augmented <lean_path> <project_root> <out_dir>

Prints a single JSON object to stdout and exits. No other stdout output.
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid as uuid_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import leanclient as lc

from rwens.canonicalization.renaming import VariableRenamer
from rwens.logger import get_logger, set_log_file
from rwens.utils.applier import StateFetchAbort
from rwens.multithreading.utils import run_with_timeout
from rwens.utils.prefix_preprocessing import fix_indentation
from rwens.utils.splitter import TacticSplitter
from rwens.utils.state import extract_state_text
from rwens.dataset.utils import should_discard_file, split_declarations_theorem_proof

logger = get_logger(__name__)
GOAL_TIMEOUT_SECONDS = 120.0


def _infer_uuid_from_path(p: Path) -> str:
    if p.name == "full.lean" and p.parent.name:
        return p.parent.name
    return p.stem


def _output_dir_exists(lean_path: Path, out_dir: Path) -> bool:
    """True if the output folder for this file already exists (already processed)."""
    return (out_dir / _infer_uuid_from_path(lean_path)).is_dir()


def _get_state_at_line(
    sfc: lc.SingleFileClient, line_number: int, timeout: float = GOAL_TIMEOUT_SECONDS
) -> Optional[str]:
    """
    Get state at a specific line number.
    Returns state text on success; None on timeout; "" on error.
    """
    result, timed_out, error = run_with_timeout(
        lambda: sfc.get_goal(line=line_number, character=0),
        timeout=timeout,
    )
    if timed_out:
        return None
    if error is not None or result is None:
        return ""
    return extract_state_text(result)


def _get_states_at_lines(
    sfc: lc.SingleFileClient,
    line_numbers: List[int],
) -> Dict[int, Optional[str]]:
    """Get state at each line number sequentially."""
    results: Dict[int, Optional[str]] = {}
    for ln in line_numbers:
        results[ln] = _get_state_at_line(sfc, ln)
    return results


def process_one_file(
    *,
    lean_path: Path,
    project_root: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Process a single full.lean file and generate augmented dataset entries.

    Returns:
        JSON-serializable dict: {"ok": True, "uuid": str, "saved": int} on success,
        or {"ok": False, "uuid": str, "error": str} on failure.
    """
    uuid = _infer_uuid_from_path(lean_path)
    project_path = Path(project_root)

    if _output_dir_exists(lean_path, out_dir):
        return {"ok": True, "uuid": uuid, "saved": 0}

    try:
        full_text = lean_path.read_text(encoding="utf-8")
        full_text = fix_indentation(full_text)
    except Exception as e:
        return {"ok": False, "uuid": uuid, "error": repr(e)}

    try:
        decls, theorem_stmt, _ = split_declarations_theorem_proof(full_text)
    except ValueError as e:
        return {"ok": False, "uuid": uuid, "error": repr(e)}

    tactics = TacticSplitter.split_proof_into_tactics(full_text)
    if not tactics:
        return {"ok": True, "uuid": uuid, "saved": 0}

    # If there are more than 1 theorems in the file, discard it
    # while creating an empty folder to show that it's been processed
    if should_discard_file(full_text):
        (out_dir / uuid).mkdir(parents=True, exist_ok=True)
        return {"ok": True, "uuid": uuid, "saved": 0}

    proof_start_idx, _ = TacticSplitter._find_proof_lines(full_text)
    if proof_start_idx == -1:
        return {"ok": True, "uuid": uuid, "saved": 0}

    state_query_lines: List[int] = []
    for line_id, _ in tactics:
        state_query_lines.append(proof_start_idx + 1 + line_id)

    set_log_file(out_dir / "rwens.log")

    client = lc.LeanLSPClient(project_root, prevent_cache_get=True)
    temp_file: Optional[Path] = None

    try:
        temp_file = project_path / f".temp/_aug_{uuid_module.uuid4().hex[:8]}.lean"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file.write_text(full_text, encoding="utf-8")

        rel_path = temp_file.relative_to(project_path).as_posix()
        client.open_file(rel_path)
        sfc = client.create_file_client(rel_path)

        states_at_lines = _get_states_at_lines(sfc, state_query_lines)

        for _line, state in states_at_lines.items():
            if state is None:
                return {
                    "ok": False,
                    "uuid": uuid,
                    "error": "state fetch timed out",
                }

        valid_tactics: List[Tuple[int, int, str, Optional[str]]] = []
        for idx, (line_id, tactic_text) in enumerate(tactics):
            state_line = state_query_lines[idx]
            state = states_at_lines.get(state_line)
            if state is None or state.strip() == "no goals":
                continue
            valid_tactics.append((idx, line_id, tactic_text, state))

        valid_tactics.sort(key=lambda x: x[0])

        if not valid_tactics:
            return {"ok": True, "uuid": uuid, "saved": 0}

        _, proof_lines = TacticSplitter._find_proof_lines(full_text)
        iterative = VariableRenamer(project_root)

        try:
            iterative.reset(decls, theorem_stmt)

            prefix_full = decls + theorem_stmt
            if not prefix_full.endswith("\n"):
                prefix_full += "\n"

            cursor = 0
            saved = 0

            for idx, line_id, tactic_text, original_state in valid_tactics:

                # Add everything before the tactic that hasn't been added yet
                block = "\n".join(proof_lines[cursor:line_id])
                if block.strip():
                    if not block.endswith("\n"):
                        block += "\n"
                    iterative.update(block)
                    prefix_full += block

                try:
                    pre_augmented_state, augmented_state, *_ = iterative.get_states()
                except StateFetchAbort as e:
                    return {"ok": False, "uuid": uuid, "error": str(e)}

                augmented_tactic = iterative._rename_tactic(tactic_text)
                prefix_at_tactic = prefix_full

                entry_file = out_dir / uuid / f"tactic_{idx:04d}.txt"
                entry_file.parent.mkdir(parents=True, exist_ok=True)
                with entry_file.open("w", encoding="utf-8") as f:
                    if original_state.strip() != pre_augmented_state.strip():
                        f.write("=" * 60 + "\n")
                        f.write(
                            f"uuid: {uuid}  tactic_idx: {idx}  line_id: {line_id}\n"
                        )
                        f.write("=" * 60 + "\n")
                        f.write("ORIGINAL_STATE (LSP):\n")
                        f.write("-" * 40 + "\n")
                        f.write(original_state.rstrip() + "\n")
                        f.write("-" * 40 + "\n")
                        f.write("PRE_AUGMENTED_STATE (variable renamer):\n")
                        f.write("-" * 40 + "\n")
                        f.write(pre_augmented_state.rstrip() + "\n")
                        f.write("-" * 40 + "\n\n")
                    else:
                        f.write(prefix_at_tactic.rstrip() + "\n")
                        f.write("========================\n")
                        f.write(original_state.rstrip() + "\n")
                        f.write("========================\n")
                        f.write(tactic_text.rstrip() + "\n")
                        f.write("========================\n")
                        f.write(augmented_state.rstrip() + "\n")
                        f.write("========================\n")
                        f.write(augmented_tactic.rstrip() + "\n")

                saved += 1
                cursor = line_id

            return {"ok": True, "uuid": uuid, "saved": saved}

        except (BrokenPipeError, ConnectionResetError):
            os._exit(1)
        except StateFetchAbort as e:
            return {"ok": False, "uuid": uuid, "error": str(e)}
        finally:
            try:
                iterative.close()
            except Exception:
                pass

    except (BrokenPipeError, ConnectionResetError):
        os._exit(1)
    except StateFetchAbort as e:
        return {"ok": False, "uuid": uuid, "error": str(e)}
    except Exception as e:
        return {"ok": False, "uuid": uuid, "error": repr(e)}
    finally:
        try:
            client.close()
        except Exception:
            pass
        if temp_file is not None and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass

    return {"ok": True, "uuid": uuid, "saved": 0}


def main() -> int:
    """Worker entrypoint: args = lean_path project_root out_dir. Prints JSON to stdout."""
    if len(sys.argv) != 4:
        out = {
            "ok": False,
            "error": "usage: python -m rwens.dataset.generate_augmented <lean_path> <project_root> <out_dir>",
        }
        print(json.dumps(out))
        sys.stdout.flush()
        return 2

    lean_path = Path(sys.argv[1]).resolve()
    project_root = sys.argv[2]
    out_dir = Path(sys.argv[3]).resolve()

    # Redirect stdout to stderr during processing so only our final JSON goes to stdout.
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        result = process_one_file(
            lean_path=lean_path,
            project_root=project_root,
            out_dir=out_dir,
        )
    except (BrokenPipeError, ConnectionResetError):
        sys.stdout = old_stdout
        out = {
            "ok": False,
            "uuid": _infer_uuid_from_path(lean_path),
            "reason": "broken_pipe",
            "error": "BrokenPipeError or ConnectionResetError",
        }
        print(json.dumps(out))
        sys.stdout.flush()
        os._exit(1)
    finally:
        sys.stdout = old_stdout

    print(json.dumps(result))
    sys.stdout.flush()
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
