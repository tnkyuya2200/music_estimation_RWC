#!/bin/bash

# $1: database file
# $2: test data dir
# $3: result data dir

echo $1
echo $2
echo $3
mkdir -p $3
for file in $2/*.wav; do
	python src/insert_test_music.py $1 $file
	python src/separate_single.py $1 0
	python src/music_estimation_single_old.py $1 $file $3
	echo estimated $file
done
