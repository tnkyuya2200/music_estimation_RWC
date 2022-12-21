#!/bin/bash -eu

# $1: database file

python src/get_all_ID.py $1 > IDlist.txt
for i in $(cat IDlist.txt); do
	python src/mysql/separate_single.py $1 $i
	echo ID:$i separated!
done
rm IDlist.txt

