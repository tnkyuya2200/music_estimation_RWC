#!/bin/bash

for file in $2/*; do
	python src/insert_test_music.py $1 $file
	python src/separate_single.py $1 0
	python src/music_estimation_single.py $1 $file
	echo estimated $file
done