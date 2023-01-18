#!/bin/bash -eu

# $1: database file
# $2: test data dir
# $3: result data dir

python src/insert_test_music.py $1 $2
python src/separate_single.py $1 0
python src/music_estimation_single.py $1 1 $3
echo estimated $file
