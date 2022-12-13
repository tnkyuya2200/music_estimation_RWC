#!/bin/bash -eu

python src/get_all_ID.py > IDlist.txt
for i in $(cat IDlist.txt); do
	python src/separate_single.py $i
	echo ID:$i separated!
done
rm IDlist.txt
