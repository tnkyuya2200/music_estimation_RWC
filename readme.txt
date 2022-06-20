init_database.bash [database file] [csv file]
	initializes database using csv file

make_testdata.bash [database file] [output directory] [start] [end]
	make testdata for datas in database file whose ID from start to end
	outputs changes to changes.csv

make_testdata_all_ID.bash [database file] [output directory]
	make testdata for datas in database file whose ID from start to end
	outputs changes to changes.csv

analyze.bash [database file] [start] [end]
	analyze musics from start to end

analyze_all_ID.bash [database file]
	analyze musics in database file

music_estimation.bash [database file] [directory name] [output dir]
	recognizes musics in directory name using database file

all_jobs.bash [database file] [csv file] [directory name] [output dir]
	do all job series
music.db has 278 datas
