#!/usr/bin/env bash
hf_path=${1}
tf_path="/home/patrick/t5/models/temp"

python /home/patrick/transformers/src/transformers/models/t5/convert_t5_original_tf_checkpoint_to_pytorch.py --tf_checkpoint_path "${tf_path}" --config_file "${hf_path}/config.json" --pytorch_dump_path "${hf_path}"
