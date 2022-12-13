#!/bin/bash -eu

# $1 - csv file
# $2 - test datas
# $3 - result dir

bash init_database.bash $1
bash separate_all_ID.bash
bash analyze_all_ID.bash
bash make_testdata_all_ID.bash $2
bash music_estimation.bash $2 $3
