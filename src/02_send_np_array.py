# point to point communication
# numpy array fast way!

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = np.arange(100, dtype='i')
    req = comm.Isend([data, MPI.INT], dest=1, tag=77)
    req.wait()
elif rank == 1:
    data = np.empty(100, dtype='i')
    req = comm.Irecv([data, MPI.INT], source=0, tag=77)
    req.wait()
    print(data)
