for %%i in (%2, 1, %3) do (
	python src/separate_single.py %1 %%i
	python src/analyze_single.py %1 %%i
)