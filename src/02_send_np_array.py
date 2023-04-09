# point to point communication
# non-blocking send an recieve arrays between two processes

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n = 5
if rank == 0:
    data = np.arange(n, dtype='i')
    req = comm.Isend([data, MPI.INT], dest=1, tag=77)
    req.wait()
elif rank == 1:
    data = np.empty(n, dtype='i')
    req = comm.Irecv([data, MPI.INT], source=0, tag=77)
    req.wait()
    print(data)

# execute: mpiexec -n 2 python 02_send_np_array.py
# output: [0 1 2 3 4]

