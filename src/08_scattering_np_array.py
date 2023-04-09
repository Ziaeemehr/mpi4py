# Scattering numpy array
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sendbuf = None
if rank == 0:
    sendbuf = np.empty([size, 5], dtype='i')
    sendbuf.T[:,:] = range(size) 
    print(f"rank : {rank} \n ", sendbuf)
    # 0 [[0 0 0 0 0]  each row fillied with rank
    #   [1 1 1 1 1]]
recvbuf = np.empty(5, dtype='i')
comm.Scatter(sendbuf, recvbuf, root=0)
assert(np.allclose(recvbuf, rank))

# execute: mpiexec -n 4 python 08_scattering_np_array.py
