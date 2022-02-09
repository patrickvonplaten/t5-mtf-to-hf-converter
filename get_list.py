#!/usr/bin/env python3
import sys
from huggingface_hub import HfApi

tag_name = str(sys.argv[1])

api = HfApi()


def get_model_ids(filter_tag=None):
    models = api.list_models(filter=filter_tag)
    model_ids = [x.modelId for x in models]
    model_ids = [x.split("/")[-1] for x in model_ids if x.split("/")[0] == "NewT5"]
    return model_ids


all_model_list = get_model_ids()
successful_model_list = get_model_ids("t5-new-success")
incorrect_model_list = list(set(all_model_list) - set(successful_model_list))

print("=" * 20 + " NEED FIXING " + "=" * 20)
print("\n".join(incorrect_model_list))
print("=" * 50)
print("\n\n")

model_list = get_model_ids(filter_tag=tag_name)

print("=" * 20 + f"{tag_name}" + "=" * 20)
print("\n".join(model_list))
print("=" * 50)

mapping = {}
with open("./hf_names_dict_reversed.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        value, key = line.strip().split()
        mapping[key] = value

with open("./new_list_temp.txt", "w") as f:
    for line in model_list:
        f.write(f"{mapping[line]} {line}\n")
