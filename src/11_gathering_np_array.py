from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


n = 5
sendfub = np.zeros(n, dtype=np.float32) + rank
recvbuf = None 

if rank == 0:
    recvbuf = np.empty((size, n), dtype=np.float32)
comm.Gather(sendfub, recvbuf, root=0)
if rank ==0:
    for i in range(size):
        assert(np.allclose(recvbuf[i, :], i))

