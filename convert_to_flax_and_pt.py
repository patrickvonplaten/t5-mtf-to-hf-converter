#!/usr/bin/env python3
from transformers import FlaxT5ForConditionalGeneration, TFT5ForConditionalGeneration, T5TokenizerFast
import sys

hf_checkpoint = str(sys.argv[1])

model = FlaxT5ForConditionalGeneration.from_pretrained(hf_checkpoint, from_pt=True)
model.save_pretrained(hf_checkpoint)

model = TFT5ForConditionalGeneration.from_pretrained(hf_checkpoint, from_pt=True)
model.save_pretrained(hf_checkpoint)

tok = T5TokenizerFast.from_pretrained(hf_checkpoint)
tok.save_pretrained(hf_checkpoint)
