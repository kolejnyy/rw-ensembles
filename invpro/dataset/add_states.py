"""
Process whole proofs into tuples (prefix, state, next_tactic).

This module processes Lean proof files by:
1. Removing comments
2. Extracting states for each line using Lean LSP
3. Creating tuples of (prefix, state, next_tactic) for training
"""

import argparse
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import leanclient as lc
from tqdm import tqdm

from invpro.dataset.utils import (
    extract_state_text,
    remove_comments,
    split_into_lines_with_tactics,
)
from invpro.multithreading.utils import run_with_timeout

# Timeout for each `get_goal` call (in seconds)
GOAL_TIMEOUT_SECONDS = 90.0



def process_file(
    project_root: Path,
    input_file: Path,
    output_dir: Path,
    processed: set[str],
    processed_lock: threading.Lock,
    force: bool,
) -> tuple[str, bool]:
    """
    Process a single proof file by extracting states for each line.
    
    Parameters
    ----------
    project_root:
        Root of the Lean project (where `lakefile.lean` or `lakefile.toml` lives).
    input_file:
        Path to the proof file to process.
    output_dir:
        Directory where output files will be saved.
    processed:
        Shared set of already-processed file IDs.
    processed_lock:
        Lock protecting access to `processed`.
    force:
        If True, recompute states even if already processed.
        
    Returns
    -------
    (file_key, success):
        file_key is the file name; success is True if the file was processed.
    """
    file_key = input_file.stem  # filename without extension
    
    # Quick check without lock
    with processed_lock:
        if file_key in processed and not force:
            return file_key, False
    
    if not input_file.exists():
        return file_key, False
    
    # Read and clean the proof file
    proof_text = input_file.read_text(encoding="utf-8")
    proof_text = remove_comments(proof_text)
    
    # Split into (prefix, next_tactic) pairs
    prefix_tactic_pairs = split_into_lines_with_tactics(proof_text)
    
    if not prefix_tactic_pairs:
        return file_key, False
    
    # Create output file for this proof
    output_file = output_dir / f"{file_key}.jsonl"
    
    # Check if already processed
    if output_file.exists() and not force:
        with processed_lock:
            processed.add(file_key)
        return file_key, False
    
    # Create a client for this worker thread
    client = lc.LeanLSPClient(str(project_root), prevent_cache_get=True)
    temp_file = None
    try:
        # Write cleaned proof to a temporary file in the project root
        # so LSP can access it
        temp_file = project_root / f"_temp_{file_key}.lean"
        temp_file.write_text(proof_text, encoding="utf-8")
        
        rel_path = temp_file.relative_to(project_root).as_posix()
        sfc = client.create_file_client(rel_path)
        
        # Process each tactic
        results = []
        proof_started = False  # Track if we've reached the start of the actual proof
        
        for tactic_idx, (prefix, next_tactic) in enumerate(prefix_tactic_pairs):
            if not next_tactic.strip():  # Skip empty tactics
                continue
            
            # Check if this prefix ends with ":= by" (start of actual proof)
            # We need to check the last line of the prefix (after stripping trailing whitespace)
            if not proof_started:
                # Get the last non-empty line of the prefix
                prefix_lines = [line for line in prefix.split('\n') if line.strip()]
                if prefix_lines:
                    last_line = prefix_lines[-1].rstrip()
                    if last_line.endswith(":= by"):
                        proof_started = True
                    else:
                        # Skip entries before the proof starts (imports, problem definition, etc.)
                        continue
                else:
                    # Empty prefix, skip
                    continue
            
            # Count lines in prefix to get where the tactic starts
            # Handle empty prefix case: if prefix is empty, split returns [''] with len=1
            if prefix.strip():
                prefix_lines = prefix.split('\n')
                tactic_start_line_0idx = len(prefix_lines)
            else:
                tactic_start_line_0idx = 0
            
            # Count lines in the tactic (may be multiple lines for multi-line tactics)
            tactic_lines = next_tactic.split('\n')
            tactic_line_count = len(tactic_lines)
            
            # Get the state at the END of the tactic (after it's been applied)
            # LSP uses 1-indexed line numbers
            # If tactic starts at line N (0-indexed) and has M lines, it ends at line N+M-1 (0-indexed)
            # In 1-indexed: line N+M
            line_number = tactic_start_line_0idx + tactic_line_count
            
            # Get goal state at this line (after the tactic is complete)
            result, timed_out, error = run_with_timeout(
                lambda: sfc.get_goal(line=line_number, character=0),
                timeout=GOAL_TIMEOUT_SECONDS,
            )
            
            if timed_out:
                print(
                    f"[add_states] [{file_key}]   get_goal timed out after "
                    f"{GOAL_TIMEOUT_SECONDS:.1f}s at line {line_number}; skipping remaining lines"
                )
                break
            
            if error is not None:
                print(
                    f"[add_states] [{file_key}]   Error getting goal at line {line_number}: {error}"
                )
                # Continue processing other lines even if one fails
                continue
            
            state_text = extract_state_text(result)
            
            # Store the tuple (prefix, state, next_tactic)
            results.append({
                "tactic_idx": tactic_idx,
                "tactic_start_line": tactic_start_line_0idx + 1,  # 1-indexed
                "tactic_end_line": line_number,  # 1-indexed
                "prefix": prefix,
                "state": state_text,
                "next_tactic": next_tactic,
            })
        
        # Write results to JSONL file
        import json
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        with processed_lock:
            processed.add(file_key)
        
        return file_key, True
    finally:
        try:
            client.close()
        except Exception as e:
            print(f"[add_states] [{file_key}] Warning: error while closing LeanLSPClient: {e}")
        # Clean up temporary file
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception as e:
                print(f"[add_states] [{file_key}] Warning: error removing temp file: {e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract states from Lean proofs and create (prefix, state, next_tactic) tuples."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Lean project root (where lakefile.lean or lakefile.toml lives).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing proof files to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where output JSONL files will be saved.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute states even if output files already exist.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads for parallel processing (default: number of CPU cores).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of files to process (default: no limit).",
    )
    return parser


def run_cli(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine number of workers (default to CPU count)
    num_workers = args.workers or os.cpu_count() or 4
    print(f"[add_states] Using {num_workers} worker threads for parallel processing")
    print(f"[add_states] Project root: {project_root}")
    print(f"[add_states] Input directory: {input_dir}")
    print(f"[add_states] Output directory: {output_dir}")
    if args.limit:
        print(f"[add_states] Processing limit: {args.limit} files")

    # Thread-safe set to track processed files
    processed: set[str] = set()
    processed_lock = threading.Lock()
    
    # Thread-safe counter for successfully processed files
    success_count = [0]  # Use list to allow modification in nested functions
    success_lock = threading.Lock()
    
    # Flag to signal shutdown
    shutdown_flag = threading.Event()
    
    def signal_handler(signum, frame):
        print("\n[add_states] Caught interrupt signal, shutting down...")
        shutdown_flag.set()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGBREAK'):  # Windows
        signal.signal(signal.SIGBREAK, signal_handler)

    executor = ThreadPoolExecutor(max_workers=num_workers)
    active_futures = set()
    
    try:
        while not shutdown_flag.is_set():
            # Collect all proof files
            proof_files = list(input_dir.glob("*"))
            proof_files = [f for f in proof_files if f.is_file()]
            
            if not proof_files:
                print("[add_states] No files to process, sleeping for 1 second")
                time.sleep(1.0)
                continue
            
            # Filter out already processed files (with lock)
            with processed_lock:
                pending_files = [
                    f for f in proof_files
                    if f.stem not in processed or args.force
                ]
            
            if not pending_files:
                print("[add_states] All files processed, sleeping for 1 second")
                time.sleep(1.0)
                continue
            
            # Check if we've reached the limit
            with success_lock:
                if args.limit and success_count[0] >= args.limit:
                    print(f"[add_states] Reached processing limit of {args.limit} files")
                    shutdown_flag.set()
                    break
            
            # Limit the number of files to submit if we have a limit
            if args.limit:
                with success_lock:
                    remaining = args.limit - success_count[0]
                    if remaining <= 0:
                        print(f"[add_states] Reached processing limit of {args.limit} files")
                        shutdown_flag.set()
                        break
                    # Only submit up to the remaining limit
                    pending_files = pending_files[:remaining]
            
            # Submit pending files to the executor
            print(f"[add_states] Found {len(pending_files)} files to process")
            
            for proof_file in pending_files:
                if shutdown_flag.is_set():
                    break
                
                # Double-check limit before submitting
                if args.limit:
                    with success_lock:
                        if success_count[0] >= args.limit:
                            break
                    
                future = executor.submit(
                    process_file,
                    project_root=project_root,
                    input_file=proof_file,
                    output_dir=output_dir,
                    processed=processed,
                    processed_lock=processed_lock,
                    force=args.force,
                )
                active_futures.add(future)
            
            # Process completed futures while remaining responsive to shutdown
            completed_count = 0
            with tqdm(total=len(active_futures), desc="Processing", unit="file") as pbar:
                while active_futures and not shutdown_flag.is_set():
                    done_futures = set()
                    
                    for future in list(active_futures):
                        # Non-blocking check if future is done
                        if future.done():
                            try:
                                file_key, success = future.result()
                                if success:
                                    completed_count += 1
                                    # Update success count if limit is set
                                    if args.limit:
                                        with success_lock:
                                            success_count[0] += 1
                                            if success_count[0] >= args.limit:
                                                print(f"\n[add_states] Reached processing limit of {args.limit} files")
                                                shutdown_flag.set()
                                pbar.update(1)
                            except Exception as e:
                                print(f"\n[add_states] Error processing file: {e}")
                                pbar.update(1)
                            done_futures.add(future)
                    
                    # Remove completed futures
                    active_futures -= done_futures
                    
                    # Small sleep to avoid busy waiting
                    if active_futures:
                        time.sleep(0.1)
                
                # If shutdown was requested, cancel all remaining futures
                if shutdown_flag.is_set():
                    print(f"\n[add_states] Cancelling {len(active_futures)} remaining futures...")
                    for future in active_futures:
                        future.cancel()
                    active_futures.clear()
                    break
            
            print(f"[add_states] Completed {completed_count} files in this batch")
            
            # Check if we've reached the limit after processing this batch
            if args.limit:
                with success_lock:
                    if success_count[0] >= args.limit:
                        print(f"[add_states] Reached processing limit of {args.limit} files. Stopping.")
                        shutdown_flag.set()
                        break
            
            # Small delay before next iteration
            if not shutdown_flag.is_set():
                time.sleep(0.1)
    
    finally:
        print("[add_states] Shutting down executor...")
        executor.shutdown(wait=False, cancel_futures=True)
        print("[add_states] Shutdown complete")


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
