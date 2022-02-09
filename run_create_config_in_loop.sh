#!/usr/bin/env bash
while read tf_name hf_name ; do
    echo "Load: ${tf_name} and create: ${hf_name}..."
	echo "======================================================================"
	bash ./create_repo.sh ${tf_name} ${hf_name}
done < "./hf_names_dict.txt"
