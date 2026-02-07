#!/bin/bash

# Read npz file paths from file.txt (space-separated)
while IFS=' ' read -r -a npz_files; do
    for npz in "${npz_files[@]}"; do
        if [ -n "$npz" ]; then
            echo "Processing: $npz"
            ./inference_onnx --input "$npz" --onnx ../U-SAM/u_sam.onnx
            if [ $? -ne 0 ]; then
                echo "Error processing $npz"
            fi
        fi
    done
done < file.txt