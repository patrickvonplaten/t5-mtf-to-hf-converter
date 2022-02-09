#!/usr/bin/env bash
while read tf_name hf_name ; do
    echo "Load: ${tf_name} and create: ${hf_name}..."
	echo "======================================================================"
	bash ./download_t5.sh ${hf_name} ${tf_name} 
done < "./new_list.txt"
