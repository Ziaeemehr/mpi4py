# Collective Communication
# scattering data

from collections import OrderedDict
from mpi4py import MPI
import numpy as np 

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    data = [{"a": np.ones(5) * 2}, {"b": np.ones(5) * 3}]
else:
    data = None

data = comm.scatter(data, root=0)
print("Rank {}: {}".format(rank, data))

