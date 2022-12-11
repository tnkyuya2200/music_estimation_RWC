#!/bin/bash -eu

# $1 - database file
# $2 - csv file
# $3 - test datas
# $4 - result dir

rm $1
rm $3 -r
rm $4/*

bash init_database_mysql.bash $1 $2
bash separate_all_ID_mysql.bash $1
bash analyze_all_ID_mysql.bash $1
bash make_testdata_all_ID_mysql.bash $1 $3
bash music_estimation_mysql.bash $1 $3 $4
