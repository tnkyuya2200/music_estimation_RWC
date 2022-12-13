#!/bin/bash -eu

# $1: test data dir
 
mkdir -p $1/

python src/get_all_ID.py > IDlist.txt
for i in $(cat IDlist.txt); do
	python src/make_testdata_single.py $1 $i >> $1/changes.csv
	echo ID:$i testdata created!
done
rm IDlist.txt
