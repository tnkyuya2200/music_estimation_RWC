1. Environment Preparation

Install anaconda and create an environment with the following command:
$ conda env create -f env_py38.yml

Activate the environment with the following command:
$ conda activate py38_ME


2. Prepare database

Make a no-header CSV file with the following format

One line represents one song, and consists of the following:
Music ID(must be unique), No., Composer(Japanese), Composer(English), Artist(Japanese), Artist(English), Title(Japanese), Title(English), 
CD_Number, Track Number, Genre(Japanese), Genre(English), Sub Genre(Japanese), Sub Genre(English), FilePath

Music ID and FilePath must not be blank, but you can enter blank to the others.

Then execute the following command:
$ bash init_database.bash [database file] [csv file]

This initializes database [database file] with [csv file]

3. Analyze the music

	Execute the following command:
	$ bash separate_all_ID.bash [database file]
		This command separates music in the [database file].
		Results are stored in the database.

	$ bash analyze_all_ID.bash	[database file]
		This command analyzes all music in the [database file].
		Results are stored in the database.

	Notice You have to do the job in the order.

4. Prepare test data

	4.1 Prepare test data with the command
	Execute the following command:
	$	bash make_testdata_all_ID.bash [database file] [output directory]

	This makes test data for all songs in [database file] and outputs to [output directory]
	It creates five types of test data.
	- raw: The music itself in the database
	- noise: The music with noise
	- pitch: The music transposed in ±5 (excludes 0)
	- speed: The speed of music changed between ×0.50 to ×1.50 (excludes ×1)
	- snipped: The snipped music lasts 1 minute to whichever shorter 180 seconds or 0.8 times the length of the music

	4.2 Prepare test data with your own
	You may prepare the test data on your own.
	Put the song in a directory.


5. Recognize Music

	Execute the following command:
	$ bash music_estimation.bash [database file] [directory name] [output dir]
	This command recognizes music in [directory name] using the [database file] and outputs results in [output dir].
	Once analyzed, the file name of the test data changes from ".wav" to ".wav.analyzed", so if you re-execute, the command ignores the already analyzed file.
	Rename the file ".wav.analyzed" to ".wav" to re-analyze.

	The result is a JSON file with following format:
	- "test_file" text: the file name of the test data
	- "db" list: a list of each score of the database music
		- "ID" int: database song ID
		- "sim" dict: similarity scores
			- "vocal" float: vocal similarity score
			- "chords" float: chord progression similarity score
			- "average" float: average score of vocal similarity score and chord progression similarity score



The following command executes all the jobs.
$ bash all_jobs.bash [database file] [csv file] [test data dir] [output dir]

6. Recognition rate
	You can see the recognition rate in the following command:
	$ python rank.py [result dir]

You may get this project by:
$ git clone https://github.com/tnkyuya2200/music_estimation_RWC

git is hard