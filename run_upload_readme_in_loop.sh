#!/usr/bin/env bash
cur_path=$(pwd)
while read a hf_name ; do
    echo "Upload: ${hf_name}..."
	rm -rf ${hf_name}

	git lfs install
	GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/NewT5/${hf_name}

	python /home/patrick/t5/models/README.py ${hf_name}

	cd ${hf_name}

	git add . && git commit -m "Upload README.md" && git push

	cd ${cur_path}

	rm -rf ${hf_name}
	echo "======================================================================"
done < "./new_list.txt"
