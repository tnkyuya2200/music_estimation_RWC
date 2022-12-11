#!/bin/bash -eu

# $1 - database file
# $2 - csv file
# $3 - test datas
# $4 - result dir

rm $1
rm $4/*

bash init_database.bash $1 $2
bash separate_all_ID.bash $1
bash analyze_all_ID.bash $1
bash make_testdata_all_ID.bash $1 $3
bash music_estimation.bash $1 $3 $4
