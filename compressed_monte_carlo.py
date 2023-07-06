import numpy as np
import scipy.stats as stats
import multiprocessing

# This is the function that each worker will execute
def _sample_worker(args):
    num_samples, target_dist, compress_func = args
    samples = []
    while len(samples) < num_samples:
        # generate a sample from the target distribution
        sample = target_dist.rvs()
        # apply the compression function
        compressed_sample = compress_func(sample)
        samples.append(compressed_sample)
    return samples

# This is the compression function defined as a proper function
def compress_func(x):
    return x**2

class CompressedMonteCarlo:
    def __init__(self, target_dist, compress_func, num_samples, num_workers=1):
        self.target_dist = target_dist
        self.compress_func = compress_func
        self.num_samples = num_samples
        self.num_workers = num_workers

    def sample(self):
        pool = multiprocessing.Pool(processes=self.num_workers)
        worker_args = [(self.num_samples // self.num_workers, self.target_dist, self.compress_func)] * self.num_workers
        samples = pool.map(_sample_worker, worker_args)
        samples = np.concatenate(samples)
        pool.close()
        pool.join()
        return samples



if __name__ == "__main__":
    multiprocessing.freeze_support()
    # define target distribution
    target_dist = stats.norm(loc=0, scale=1)  # Normal distribution for example
    # create and run a Compressed Monte Carlo sampler
    cmc = CompressedMonteCarlo(target_dist, compress_func, num_samples=10000, num_workers=4)
    samples = cmc.sample()
    print(samples)

