import time
import random

def heartbeat_simulator(state='rest'):
    bpm = {'rest': 60, 'active': 100, 'effort': 130}

    while True:
        current_bpm = bpm[state]
        current_bpm += current_bpm * random.gauss(0, 0.1)  # add variability with normal distribution
        timestamp = int(time.time() * 1000)  # Unix timestamp in milliseconds
        yield {'timestamp': timestamp, 'heartbeat': round(current_bpm)}

# Usage:
rest_gen = heartbeat_simulator(state='rest')
active_gen = heartbeat_simulator(state='active')
effort_gen = heartbeat_simulator(state='effort')

for _ in range(10):
    print(next(rest_gen))
    print(next(active_gen))
    print(next(effort_gen))
    time.sleep(1)  # simulate a delay of 1 second
