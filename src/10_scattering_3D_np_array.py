from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sendbuf = None
if rank == 0:
    sendbuf = np.empty((size, 2, 3), dtype=np.float32)
    for i in range(size):
        sendbuf[i, :, :] = i
recvbuf = np.empty((2, 3), dtype=np.float32)
comm.Scatter(sendbuf, recvbuf, root=0)
assert(np.allclose(recvbuf, rank * np.ones((2, 3), dtype=np.float32)))