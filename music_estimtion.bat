for /d %%d in (%1/*) do (
	./src/make_testdata_single.py %1 file
	./src/separete_single.py %1 0
	./src/analyze_single.py %1 0
	./src/music_estimation_single.py %1
)
