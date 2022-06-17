#!/bin/bash

for ((i=1; i<=$2; i++)); do
	python re_separate_single.py $1 $i
	python re_analyze_single.py $1 $i