#!/usr/bin/env bash
hf_name=${1}
tf_path=${2}
hf_path="/home/patrick/t5/models/hf_models/${hf_name}"
current_path="/home/patrick/t5/models"

temp_folder="/home/patrick/t5/models/temp"
num_of_checks=4

rm -rf ${temp_folder}
rm -rf ${hf_path}

cd "${current_path}/hf_models"
git lfs install
git clone https://huggingface.co/NewT5/${hf_name}
cd ${current_path}

mkdir ${temp_folder}
cd ${temp_folder}

gsutil cp gs://scenic-bucket/scaling_explorer/scaling_explorer/${tf_path}/operative_config.gin .
gsutil cp gs://scenic-bucket/scaling_explorer/scaling_explorer/${tf_path}/checkpoint .
gsutil cp gs://scenic-bucket/scaling_explorer/scaling_explorer/${tf_path}/model.ckpt-524288.index .
gsutil cp gs://scenic-bucket/scaling_explorer/scaling_explorer/${tf_path}/model.ckpt-524288.meta .

# single digit num checkpoitns
for ((i = 0 ; i < ${num_of_checks} ; i++)); do
	gsutil cp gs://scenic-bucket/scaling_explorer/scaling_explorer/${tf_path}/model.ckpt-524288.data-0000${i}-of-0000${num_of_checks} .
done

# double digit num checkpoitns
#for ((i = 0 ; i < ${num_of_checks} ; i++)); do
#	gsutil cp gs://scenic-bucket/scaling_explorer/scaling_explorer/${tf_path}/model.ckpt-524288.data-0000${i}-of-000${num_of_checks} .
#done
#
#for ((i = 0 ; i < ${num_of_checks} ; i++)); do
#	gsutil cp gs://scenic-bucket/scaling_explorer/scaling_explorer/${tf_path}/model.ckpt-524288.data-000${i}-of-000${num_of_checks} .
#done

python /home/patrick/transformers/src/transformers/models/t5/convert_t5_original_tf_checkpoint_to_pytorch.py --tf_checkpoint_path "${temp_folder}" --config_file "${hf_path}/config.json" --pytorch_dump_path "${hf_path}"

python /home/patrick/t5/models/run.py "${hf_path}" "${temp_folder}"

cd ${hf_path}

huggingface-cli lfs-enable-largefiles ${hf_path}
git add . && git commit -m "add PyTorch model and README" && git push

rm -rf ${hf_path}

# double digit num checkpoitns
#for ((i = 0 ; i < ${num_of_checks} ; i++)); do
#	gsutil cp gs://scenic-bucket/scaling_explorer/scaling_explorer/${path}/model.ckpt-524288.data-0000${i}-of-000${num_of_checks} .
#done
#
#for ((i = 0 ; i < ${num_of_checks} ; i++)); do
#	gsutil cp gs://scenic-bucket/scaling_explorer/scaling_explorer/${path}/model.ckpt-524288.data-000${i}-of-000${num_of_checks} .
#done
