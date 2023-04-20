import numpy as np
from mpi4py import MPI
import multiprocessing as mp

SEED = 42
np.random.seed(SEED)

def single(theta_i):
    x = 2 * theta_i
    return x

def batch_run(theta):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    with mp.Pool(processes=2) as pool:
        results = pool.map(single, theta)
    return np.array(results), rank

theta = np.random.rand(10, 2)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
sendbuf = None

if rank == 0:
    sendbuf = np.array_split(theta, size, axis=0)
    # print("theta\n", theta)

data = comm.scatter(sendbuf, root=0)
result, i = batch_run(data)
gathered = comm.gather(result, root=0)

if rank == 0:
    gathered = np.concatenate(gathered, axis=0)
    print("Rank {}, gathered: {}".format(rank, gathered.shape))
    print(gathered)

print("Rank {} {}: data {} result {}".format(rank, i, data.shape, result.shape))

