mkdir %2/noise, %2/pitch, %2/raw, %2/snipped, %2/speed
if not exist %2/changes.csv (echo ID, start_samples, end_samples, speed_change, pitch_change > %2/changes.csv)

for /l %%i in (%3, 1, %4) do (
	python src/make_testdata_single.py %1 %2 %%i >> %2/changes.csv
)