#!/bin/bash -eu

bash init_database.bash database/vocals.db RWC/Data_vocals.csv
bash separate_all_ID.bash database/vocals.db
bash analyze_all_ID.bash database/vocals.db
bash music_estimation.bash database/vocals.db testdata/test_vocals2 result/result_op

