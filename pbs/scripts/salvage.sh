#!/bin/bash
# Usage: ./salvage.sh JOB_ID [--clean]
# This script parses the output of "qstat -xf JOB_ID" and extracts:
#   - SCRATCH: the scratch directory location (e.g. /scratch.ssd/$USER/job_10302860.pbs-m1)
#   - EXEC_HOST: the hostname extracted from the "exec_host2" field (e.g. fer1.natur.cuni.cz)

# Check for JOB_ID parameter
if [ -z "$1" ]; then
    echo "Usage: $0 JOB_ID [--clean]"
    exit 1
fi

JOB_ID="$1"
CLEAN=false

# Check if the optional --clean option is provided
if [ "$2" == "--clean" ]; then
    CLEAN=true
fi

HOMEDIR="/storage/brno2/home/$USER/"

# Get the full output from qstat
QSTAT_OUTPUT=$(qstat -xf "$JOB_ID")

# Extract SCRATCH:
# We grep for "SCRATCH=" and then extract the value following the "=".
# This example assumes the first match is the desired SCRATCH directory.
SCRATCH=$(echo "$QSTAT_OUTPUT" | grep -oP 'SCRATCH=\S+' | head -n 1 | cut -d'=' -f2 | sed 's/,$//')

# Extract EXEC_HOST:
# Although there is a line "exec_host = ..." here we want the fully-qualified hostname.
# The "exec_host2" line provides that information as "exec_host2 = fer1.natur.cuni.cz:15002/4*2"
# We capture the text before the first colon.
EXEC_HOST=$(echo "$QSTAT_OUTPUT" | grep -oP 'exec_host2\s*=\s*\S+' | head -n 1 | sed -E 's/exec_host2\s*=\s*([^:]+).*/\1/')

# Print the extracted values.
echo "SCRATCH: $SCRATCH"
echo "EXEC_HOST: $EXEC_HOST"

rsync -avu $EXEC_HOST:$SCRATCH/datasets/ $HOMEDIR/datasets/
rsync -avu $EXEC_HOST:$SCRATCH/outputs/ $HOMEDIR/outputs/
rsync -avu $EXEC_HOST:$SCRATCH/artefacts/ $HOMEDIR/artefacts/
rsync -avu $EXEC_HOST:$SCRATCH/logadcomp/sempca.d/logs/ $HOMEDIR/logadcomp/sempca.d/logs/

# If --clean was specified, remove all files and directories under the SCRATCH directory on the remote EXEC_HOST.
if [ "$CLEAN" = true ]; then
    echo "Cleaning all files and directories under ${SCRATCH} on host ${EXEC_HOST}..."
    # The following command removes every file and directory inside SCRATCH while preserving the SCRATCH directory itself.
    ssh "${EXEC_HOST}" "find '${SCRATCH}' -mindepth 1 -depth -exec rm -rf {} +"
fi

