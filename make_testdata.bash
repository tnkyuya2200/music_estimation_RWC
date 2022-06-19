#!/bin/bash

mkdir -p $2
mkdir -p $2/{noise, pitch, row, snhipped, speed}
for ((i=$3; i<=$4; i++)); do
	python src/make_testdata_single.py $1 $2 $i >> $2/changes.csv
done