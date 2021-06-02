import time
import sys
from tqdm import trange


def do_something():
    time.sleep(1)

def do_another_something():
    time.sleep(1)


for i in trange(10, desc='outer loop'):
    do_something()

