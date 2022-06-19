
if not exist %2 mkdir %2/noise %2/pitch %2/raw %2/snipped %2/speed
for /l %%i in (%3, 1, %4) do (
	python src/make_testdata_single.py %1 %2 %%i >> %2/changes.csv
)