#!/bin/bash -eu

# $1: database file
# $2: test data dir
 
mkdir -p $2/

python src/get_all_ID.py $1 > IDlist.txt
for i in $(cat IDlist.txt); do
	python src/make_testdata_single.py $1 $2 $i >> $2/changes.csv
	echo ID:$i testdata created!
done
rm IDlist.txt
