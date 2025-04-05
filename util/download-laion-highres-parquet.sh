#!/bin/bash

mkdir -p laion-high-resolution && cd laion-high-resolution

for i in $(seq -f "%04g" 0 127); do
    wget --header="Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/datasets/laion/laion-high-resolution/resolve/refs%2Fconvert%2Fparquet/default/train/${i}.parquet"
done

cd ..