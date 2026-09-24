
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

def load_json_if_exists(path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as e:
            print(f"ERROR: Failed to parse JSON {path}: {e}")
            sys.exit(2)
    return {}

def is_isaac_script(path):
    """Detect if script imports omni/pxr/omni.isaac by reading first ~200 lines."""
    try:
        txt = path.read_text(errors="ignore")
    except Exception:
        return False
    checks = ["import omni", "from pxr import", "omni.isaac", "from omni import", "import pxr"]
    return any(s in txt for s in checks)

def run_script(python_exec, script_path, args, log_path, timeout=None):
    cmd = [python_exec, str(script_path)] + args
    print(f"\n>>> Running: {' '.join(cmd)}")
    print(f"    logging -> {log_path}")
    start = time.time()
    with open(log_path, "wb") as logf:
        # spawn subprocess, stream stdout/stderr to log and also print a line summary
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            while True:
                chunk = proc.stdout.readline() if proc.stdout else b""
                if chunk:
                    logf.write(chunk)
                    logf.flush()
                    # echo essential lines to console (not flooding)
                    line = chunk.decode(errors="ignore").rstrip()
                    if line:
                        print(line)
                elif proc.poll() is not None:
                    break
                # timeout check
                if timeout:
                    elapsed = time.time() - start
                    if elapsed > timeout:
                        proc.kill()
                        print(f"ERROR: script timed out after {timeout:.1f}s and was killed.")
                        return proc.returncode or 124
            proc.wait()
        except KeyboardInterrupt:
            proc.kill()
            raise
    elapsed = time.time() - start
    print(f"<<< Finished {script_path.name} (rc={proc.returncode}) in {elapsed:.1f}s")
    return proc.returncode

def verify_expected_outputs(script_name, expected_list):
    """expected_list: list of file/dir paths (absolute or relative). Return True if all exist and non-empty for dirs."""
    missing = []
    for p in expected_list:
        p = os.path.expanduser(p)
        if not os.path.isabs(p):
            # treat relative to scripts dir or cwd; keep as provided
            p = os.path.abspath(p)
        if not os.path.exists(p):
            missing.append(p)
            continue
        if os.path.isdir(p) and not any(Path(p).iterdir()):
            missing.append(p + " (empty dir)")
    return missing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts-dir", required=True, help="Directory containing .py scripts to run")
    ap.add_argument("--isaac-python", default="/home/rhutvik/isaac-sim/python.sh",
                    help="Path to Isaac Sim python launcher (python.sh). Used automatically for scripts that import omni/pxr.")
    ap.add_argument("--dataset-dir", default="", help="Optional: dataset directory to verify (special-case check)")
    ap.add_argument("--run-args-json", default="run_args.json",
                    help="JSON file (in scripts-dir or cwd) mapping script filename -> list of args")
    ap.add_argument("--expected-json", default="expected_outputs.json",
                    help="JSON file mapping script filename -> list of output paths that must exist after running")
    ap.add_argument("--timeout", type=float, default=0.0, help="Per-script timeout in seconds (0 for no timeout)")
    ap.add_argument("--order-file", default="", help="Optional file listing filenames (one per line) to force run order")
    ap.add_argument("--logs-dir", default="logs", help="Directory to write logs into (inside scripts-dir by default)")
    args = ap.parse_args()

    scripts_dir = Path(args.scripts_dir).expanduser().resolve()
    if not scripts_dir.is_dir():
        print("ERROR: scripts-dir does not exist:", scripts_dir)
        sys.exit(2)

    # load run args and expected outputs
    run_args_path = Path(args.run_args_json)
    if not run_args_path.is_absolute():
        # prefer file inside scripts_dir if exists, otherwise cwd
        cand = scripts_dir / run_args_path.name
        if cand.exists():
            run_args_path = cand
        else:
            run_args_path = Path(run_args_path.name).resolve()
    expected_path = Path(args.expected_json)
    if not expected_path.is_absolute():
        cand2 = scripts_dir / expected_path.name
        if cand2.exists():
            expected_path = cand2
        else:
            expected_path = Path(expected_path.name).resolve()

    run_args = load_json_if_exists(run_args_path)
    expected_outputs = load_json_if_exists(expected_path)

    # build file list
    py_files = sorted([p for p in scripts_dir.iterdir() if p.is_file() and p.suffix == ".py"])
    # allow order override
    if args.order_file:
        order_f = Path(args.order_file)
        if not order_f.is_absolute():
            order_f = scripts_dir / order_f
        if order_f.exists():
            listed = [l.strip() for l in order_f.read_text().splitlines() if l.strip()]
            ordered = []
            for name in listed:
                candidate = scripts_dir / name
                if candidate.exists():
                    ordered.append(candidate)
                else:
                    print(f"WARNING: ordered file {name} not found in scripts-dir; ignoring")
            # append any not in list
            for p in py_files:
                if p not in ordered:
                    ordered.append(p)
            py_files = ordered

    # prepare logs dir
    logs_dir = scripts_dir / args.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    # run loop
    for script in py_files:
        print("\n===============================================================")
        print("Next script:", script.name)
        # decide python executable
        if is_isaac_script(script):
            python_exec = args.isaac_python
            if not Path(python_exec).exists():
                print("ERROR: Isaac script detected but isaac-python not found:", python_exec)
                print("Either install Isaac or set --isaac-python to your python.sh path.")
                sys.exit(3)
            print("  -> Detected Isaac script; will run with:", python_exec)
        else:
            python_exec = sys.executable  # current Python interpreter
            print("  -> Using system python:", python_exec)

        # get args for this script (from run_args mapping)
        script_args = run_args.get(script.name, [])
        if not isinstance(script_args, list):
            print("ERROR: run_args for", script.name, "must be a list in JSON.")
            sys.exit(4)

        # get timeout
        to = None if args.timeout <= 0 else float(args.timeout)

        # prepare log file
        log_path = logs_dir / f"{script.name}.log"

        # run
        rc = run_script(python_exec, script, script_args, log_path, timeout=to)
        if rc != 0:
            print(f"\nFAILURE: {script.name} returned rc={rc}. See log: {log_path}")
            sys.exit(10 + (rc & 255))

        # verify expected outputs if any
        expected_for_script = expected_outputs.get(script.name, [])
        # special-case: if dataset-dir was provided and script is isaac_replay_recorder.py, check it
        if script.name == "isaac_replay_recorder.py" and args.dataset_dir:
            expected_for_script = expected_for_script + [args.dataset_dir]

        if expected_for_script:
            missing = verify_expected_outputs(script.name, expected_for_script)
            if missing:
                print(f"ERROR: Expected outputs missing after running {script.name}:")
                for m in missing:
                    print("   -", m)
                print("Aborting sequential run.")
                sys.exit(11)

        print(f"SUCCESS: {script.name} completed and outputs (if any) verified. Proceeding to next.")

    print("\nALL SCRIPTS COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
