#!/bin/bash

# $1: database file

python src/get_all_ID.py $1 > IDlist.txt
for i in $(cat IDlist.txt); do
	python src/separate_single.py $1 $i
	python src/analyze_single.py $1 $i
	echo ID:$i analyzed!
done
rm IDlist.txt
