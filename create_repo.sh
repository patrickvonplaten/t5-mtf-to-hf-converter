#/usr/bin/env bash
tf_name=${1}
hf_name=${2}

cd hf_models

huggingface-cli repo create ${hf_name} --organization NewT5 --yes

git clone https://huggingface.co/NewT5/${hf_name}

cd ${hf_name}

gsutil cp gs://scenic-bucket/scaling_explorer/scaling_explorer/${tf_name}/operative_config.gin .

python /home/patrick/t5/models/create_config.py "./operative_config.gin" "./"

git add . && git commit -m "Add config" && git push
