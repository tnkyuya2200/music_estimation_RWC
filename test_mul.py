import sys
import time
from concurrent.futures.process import ProcessPoolExecutor
from logging import StreamHandler, Formatter, INFO, getLogger
from random import random
from time import time


def init_logger():
	handler = StreamHandler()
	handler.setLevel(INFO)
	handler.setFormatter(Formatter("[%(asctime)s] [%(threadName)s] %(message)s"))
	logger = getLogger()
	logger.addHandler(handler)
	logger.setLevel(INFO)


def task(params):
	print("start")
	(v, num_calc) = params
	a = float(v)
	for _ in range(num_calc):
		a = pow(a, a)
	print("end")
	return a


def main():
	init_logger()

	if len(sys.argv) != 5:
		print("usage: 05_process.py max_workers chunk_size num_tasks num_calc")
		sys.exit(1)
	(max_workers, chunk_size, num_tasks, num_calc) = map(int, sys.argv[1:])

	start = time()

	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		params = map(lambda _: (random(), num_calc), range(num_tasks))
		results = executor.map(task, params, chunksize=chunk_size)
	getLogger().info(sum(results))

	getLogger().info("{:.3f}".format(time() - start))


if __name__ == "__main__":
	main()