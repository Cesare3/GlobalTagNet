import multiprocessing
from os.path import dirname
from pathlib import Path

N_CPUS = multiprocessing.cpu_count()

PROJECT_DIR = Path(dirname(__file__)).parent.parent


if __name__ == "__main__":
    import os
    print(os.path.abspath(PROJECT_DIR))
