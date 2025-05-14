"""
Script to move log files to their respective artefact directories.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>

Usage:
```
python move_to_artefacts.py <log_files> <artefact_dir>
```
"""

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description="Organize files to artefact folders.")
parser.add_argument(
    "log_files",
    type=str,
    nargs="+",
    help="Log files to be organized, can be specified as shell path globs.",
)
parser.add_argument(
    "artefact_dir",
    type=str,
    help="Directory where artefacts are stored.",
)
args = parser.parse_args()
artefact_dir = Path(args.artefact_dir)

# Check if the artefact directory exists
if not artefact_dir.exists():
    print(f"Artefact directory {artefact_dir} does not exist.")
    exit(1)

filename_pattern = re.compile(r"^.*\.[oe]\d+$")
session_id_pattern = re.compile(r"^\d{6}_\d{6}$")

job_id2session_id = {}
job_id2files = defaultdict(list)


def get_session_id(file_path):
    """
    Log format:
    250421_230111 2025-04-22 01:01:41,381 DataLoader.BGL DEBUG:
    [Session ID] [Date] [Time] [Component] [Level]: [Message]
    """
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first_word = line.split()[0]
            if session_id_pattern.match(first_word):
                return first_word
    return None


# Gather files
for filespec in args.log_files:
    if "*" in filespec or "?" in filespec:
        # If the input is a glob pattern, use glob to find matching files
        files = list(Path().glob(filespec))
    else:
        # Otherwise, treat it as a single file
        files = [Path(filespec)]

    for file_path in files:
        if not file_path.exists():
            continue
        if not filename_pattern.match(file_path.name):
            continue

        job_id = str(file_path.name).split(".")[1][1:]
        job_id2files[job_id].append(file_path)

        if job_id not in job_id2session_id:
            session_id = get_session_id(file_path)
            if session_id is None:
                continue
            job_id2session_id[job_id] = session_id

# Move files to artefact directories
for job_id, session_id in job_id2session_id.items():
    files = job_id2files[job_id]

    session_dir = artefact_dir / session_id
    if not session_dir.exists():
        print(f"Session directory {session_dir} does not exist.")
        continue

    # Inside the session dir is a subdir with dataset name
    # Inside that is the method dir
    # Inside are the run results
    datasets = list(p for p in session_dir.glob("*") if p.is_dir())
    if len(datasets) != 1:
        print(f"Session directory {session_dir} does not contain exactly one dataset.")
    else:
        session_dir = datasets[0]
        methods = list(p for p in session_dir.glob("*") if p.is_dir())
        if len(methods) != 1:
            print(
                f"Dataset directory {session_dir} does not contain exactly one method."
            )
        else:
            session_dir = methods[0]

    for file_path in files:
        target_path = session_dir / file_path.name
        shutil.move(file_path, target_path)
