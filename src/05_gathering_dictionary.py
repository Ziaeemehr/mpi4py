#!/usr/bin/env python3

import numpy as np
from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
root = 0

row = random.randint(2, 5)
col = 5

local_data = rank * np.ones((row, col))
print("rank: {}, local_array: {}".format(rank, local_data))
sendbuf = np.array(local_data)

# Collect local array sizes using the high-level mpi4py gather
sendcounts = np.array(comm.gather(local_data.size, root))

# if rank == root:
#     print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
#     recvbuf = np.empty(sum(sendcounts), dtype=int)
# else:
#     recvbuf = None

# comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=root)
# if rank == root:
#     print("Gathered array: {}".format(recvbuf))
