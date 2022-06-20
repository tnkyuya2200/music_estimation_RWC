#!/bin/bash

mkdir -p $2
echo ID, start_samples, end_samples, speed_change, pitch_change > $2/changes.csv

for ((i=$3; i<=$4; i++)); do
	python src/make_testdata_single.py $1 $2 $i >> $2/changes.csv
done
