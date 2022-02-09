#!/usr/bin/env python3
import sys
from transformers import T5Config
from collections import OrderedDict

gin_file = sys.argv[1]
hf_folder = sys.argv[2]

config = T5Config.from_pretrained("/home/patrick/t5/models/template")

with open(gin_file, "r") as f:
    lines = f.readlines()

imported_parameters = {
    "d_ff": "d_ff",
    "d_kv": "d_kv",
    "d_model": "d_model",
    "num_heads": "num_heads",
    "num_layers": "num_layers",
    "encoder/make_layer_stack.num_layers": "num_layers",
    "decoder/make_layer_stack.num_layers": "num_decoder_layers",
}
param_dict = OrderedDict()
for line in lines:
    line = line.strip()
    for key in imported_parameters.keys():
        if line.startswith(key):
            value = line.split(key)[-1].replace("=", "").replace(" ", "").replace("%", "")
            if value.isdigit():
                param_dict[key] = int(value)
            else:
                value = param_dict[value]
                param_dict[key] = value

for key in param_dict:
    setattr(config, imported_parameters[key], param_dict[key])

config.save_pretrained(hf_folder)
