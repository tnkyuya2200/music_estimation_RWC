#!/bin/bash -eu

# $1: test data dir
# $2: result data dir

mkdir -p $2
for file in $1/*.wav; do
	python src/insert_test_music.py $file
	python src/separate_single.py 0
	python src/music_estimation.py $file $2
	echo estimated $file
	mv $file $file.analyzed
done
