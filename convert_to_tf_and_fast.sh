#!/usr/bin/env bash
path=${1}

cd ${path}

python -c "from transformers import TFT5ForConditionalGeneration; model = TFT5ForConditionalGeneration.from_pretrained('./', from_pt=True); model.save_pretrained('./')"

python -c "from transformers import FlaxT5ForConditionalGeneration; model = FlaxT5ForConditionalGeneration.from_pretrained('./', from_pt=True); model.save_pretrained('./')"

python -c "from transformers import T5TokenizerFast; model = T5TokenizerFast.from_pretrained('./'); model.save_pretrained('./')"
