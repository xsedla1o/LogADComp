#!/usr/bin/env bash
# usage: bash download.sh <dataset> <download_dir>
# dataset: HDFS, BGL, Thunderbird
# download_dir: the directory to save the downloaded dataset

CHOICES="{ HDFS | BGL  | TBird | embeddings }"
if [ $# -ne 2 ]; then
    echo "Usage: $0 $CHOICES <download_dir>"
    exit 1
fi

ITEM=$1
DOWNLOAD_DIR=$(realpath "$2")

# Download HDFS if requested
if [ "$ITEM" == "HDFS" ]; then
    parent_path="${DOWNLOAD_DIR}/${ITEM}/"
    mkdir -p "$parent_path"
    cd "$parent_path" || { echo "Failed to cd into $parent_path"; exit 1; }

    # Download the dataset
    wget "https://iiis.tsinghua.edu.cn/~weixu/demobuild.zip" -P "$parent_path"
    unzip -j demobuild.zip "data/online1/lg/sorted.log.gz" -d "$parent_path"
    rm demobuild.zip
    gunzip -c sorted.log.gz > HDFS.log
    rm sorted.log.gz

    # Download annotation
    zipfile="HDFS_1.tar.gz?download=1"
    wget "https://zenodo.org/record/3227177/files/${zipfile}" -P "$parent_path"
    tar -xvzf "$zipfile" anomaly_label.csv
    rm "$zipfile"
elif [ "$ITEM" == "HDFSLogHub" ]; then
    parent_path="${DOWNLOAD_DIR}/${ITEM}/"
    mkdir -p "$parent_path"
    cd "$parent_path" || { echo "Failed to cd into $parent_path"; exit 1; }

    # Download the dataset
    zipfile="HDFS_1.tar.gz?download=1"
    wget "https://zenodo.org/record/3227177/files/${zipfile}" -P "$parent_path"
    tar -xvzf "$zipfile" anomaly_label.csv HDFS.log
    mv HDFS.log HDFSLogHub.log
    rm "$zipfile"
elif [ "$ITEM" == "BGL" ]; then
    parent_path="${DOWNLOAD_DIR}/${ITEM}/"
    mkdir -p "$parent_path"
    cd "$parent_path" || { echo "Failed to cd into $parent_path"; exit 1; }

    # Download the dataset
    wget "http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/bgl2.gz" -P "$parent_path"
    gunzip bgl2.gz
    mv bgl2 BGL.log
elif [ "$ITEM" == "TBird" ]; then
    parent_path="${DOWNLOAD_DIR}/${ITEM}/"
    mkdir -p "$parent_path"
    cd "$parent_path" || { echo "Failed to cd into $parent_path"; exit 1; }

    # Download the dataset
    wget "http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/tbird2.gz" -P "$parent_path"
    gunzip tbird2.gz
    mv tbird2 TBird.log
    head -n 20000000 TBird.log >TBird_20M.log
    mv TBird_20M.log TBird.log
elif [ "$ITEM" == "embeddings" ]; then
    parent_path="${DOWNLOAD_DIR}"
    mkdir -p "$parent_path"
    cd "$parent_path" || { echo "Failed to cd into $parent_path"; exit 1; }

    # Download the embeddings
    wget "https://nlp.stanford.edu/data/glove.6B.zip" -P "$parent_path"
    unzip glove.6B.zip glove.6B.300d.txt
    rm glove.6B.zip
else
    echo "Invalid download item: $ITEM, choose one of $CHOICES"
    exit 1
fi
