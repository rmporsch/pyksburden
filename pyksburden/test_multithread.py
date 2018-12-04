import numpy as np
import multiprocessing.dummy as mp


def dummy_function(x, y):
    return x*y

def dummy_iter(i):
    for i in range(i):
        yield (i, i)

if __name__ == '__main__':
    p = mp.Pool(1)
    itera = dummy_iter(10)
    results = p.starmap(dummy_function, itera)
    p.close()
    p.join()
    print(results)
