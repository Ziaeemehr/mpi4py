# Broadcasting a torch tensor with MPI

from mpi4py import MPI
import numpy as np
import torch

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = torch.arange(5, dtype=torch.float32)
else:
    data = None

x = comm.bcast(data, root=0)
print(f"Rank {rank}: {x}")

# execute: mpiexec -n 2 python 07_broadcasting_torch.py
# output:
# Rank 0: tensor([0., 1., 2., 3., 4.])
# Rank 1: tensor([0., 1., 2., 3., 4.])

