#!/bin/bash

#python src/get_all_ID.py $1 > IDlist.txt
mkdir -p $2/{noise,pitch,raw,snipped,speed}

python src/get_all_ID.py $1 > IDlist.txt
for i in $(cat IDlist.txt); do
	python src/make_testdata_single.py $1 $2 $i >> $2/changes.csv
done
rm IDlist.txt