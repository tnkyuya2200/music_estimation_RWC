import time, os
from concurrent.futures import ThreadPoolExecutor


def count_primes_single(num, thread_num):


def count_primes(num) -> None:
	primes = 0
	for i in range(2, num + 1):
		for j in range(2, i):
			if i % j == 0:
				break
		else:
			primes += 1
	return primes

future_list = []
with ThreadPoolExecutor() as executor:
    for i in range(os.cpu_count()):
        future = executor.submit(count_primes)
        future_list.append(future)

print([x.result() for x in future_list])