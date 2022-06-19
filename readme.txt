make_testdata.bash [database file] [output directory] [start] [end]
	make testdata for datas in database file whose ID from start to end
	outputs changes to changes.csv

init_database.bash [database file] [csv file]
	initializes database using csv file

analyze.bash [database file] [start] [end]
	analyze musics from start to end

music_estimation.bash [database file] [directory name]
	recognizes musics in directory name using database file

music.db has 278 datas