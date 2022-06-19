#!/bin/bash

python src/get_all_ID.py $1 > IDlist.txt
for i in $(cat IDlist.txt); do
	python src/separate_single.py $1 $i
	python src/analyze_single.py $1 $i
done
rm IDlist.txt