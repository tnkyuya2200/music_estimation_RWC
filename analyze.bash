#!/bin/bash

for ((i=$2; i<=$3; i++)); do
	python src/separate_single.py $1 $i
	python src/analyze_single.py $1 $i
done