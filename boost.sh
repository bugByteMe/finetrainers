#!/bin/bash

ROOT=./demo2/data_sled
NUM_COPY=1
# For each video with prefix split_video_part_ under the directory $ROOT:
# Make a copy of the file
for file in $ROOT/split_video_part_*.mp4; do
    for i in $(seq 1 $NUM_COPY); do
        # Create a new file name with a suffix
        new_file="${file%.mp4}_copy_${i}.mp4"
        # Copy the file to the new file name
        cp "$file" "$new_file"
        # Copy the corresponding text file
        text_file="${file%.mp4}.txt"
        new_text_file="${text_file%.txt}_copy_${i}.txt"
        cp "$text_file" "$new_text_file"
    done
done
    