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

