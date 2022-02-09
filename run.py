#!/usr/bin/env python3
# pip install t5==0.9.3
from seqio import SentencePieceVocabulary
import os
from t5.models import MtfModel

from transformers import T5ForConditionalGeneration, T5Tokenizer

import sys

path_to_hf_checkpoint = str(sys.argv[1])
path_to_mtf_small_t5_v1_1_checkpoint = str(sys.argv[2])

path_to_mtf_small_spm_model_path = "/home/patrick/t5/models/spiece.model"

input_str = "Hello there this is an example input"
label_str = "Hi I am the possible output of the input"

score = ""
hf_score = ""
result_tag = None

with open(os.path.join(path_to_mtf_small_t5_v1_1_checkpoint, "operative_config.gin"), "r") as f:
    lines = f.readlines()
    
with open(os.path.join(path_to_mtf_small_t5_v1_1_checkpoint, "operative_config.gin"), "w") as f:
    for line in lines:
        if "VocabEmbedding" not in line:
            f.write(line)

try:
    t5_model = MtfModel(model_dir=path_to_mtf_small_t5_v1_1_checkpoint, batch_size=1, tpu=None)
    vocab = SentencePieceVocabulary(path_to_mtf_small_spm_model_path, extra_ids=100)

    # adapt original tf-mesh code so that save_example_text=False
    score = t5_model.score(inputs=[input_str], targets=[label_str], vocabulary=vocab)
    score = score[0]
except:
    result_tag = "mtf-not-loaded"

if result_tag is None:
    try:
        model = T5ForConditionalGeneration.from_pretrained(path_to_hf_checkpoint)
        tokenizer = T5Tokenizer.from_pretrained(path_to_mtf_small_spm_model_path)
        tokenizer.save_pretrained(path_to_hf_checkpoint)

        input_ids = tokenizer(input_str, return_tensors="pt").input_ids
        labels = tokenizer(label_str, return_tensors="pt").input_ids

        loss = model(input_ids, labels=labels).loss
        hf_score = -(labels.shape[-1] * loss.item())

        if abs(float(score) - float(hf_score)) > 0.01:
            result_tag = "failed"
        else:
            result_tag = "success"
    except:
        result_tag = "hf-not-loaded"

with open(os.path.join(path_to_hf_checkpoint, "README.md"), "w") as f:
    string = f"""---
tags:
- t5-new-{result_tag}
---

# Test
Hf T5: {hf_score}
MTF T5: {score}
"""
    f.write(string)
