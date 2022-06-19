#!/bin/bash

# $1 - database file
# $2 - csv file
# $3 - test data dir


bash init_database.bash $1 $2
bash make_testdata_all_ID.bash $1 $3
bash analyze_all_ID.bash $1
bash music_estimation.bash $1 $3/noise
bash music_estimation.bash $1 $3/pitch
bash music_estimation.bash $1 $3/raw
bash music_estimation.bash $1 $3/snipped
bash music_estimation.bash $1 $3/speed