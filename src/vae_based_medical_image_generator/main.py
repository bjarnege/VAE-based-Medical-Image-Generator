print("Test")

from tqdm import tqdm

import time
for i1 in tqdm(range(5)):
    for i2 in tqdm(range(300), leave=False):
        # do something, e.g. sleep
        time.sleep(0.01)