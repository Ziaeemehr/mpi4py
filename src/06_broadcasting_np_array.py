from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = np.arange(100, dtype='i')
else:
    data = np.empty(100, dtype='i')

comm.Bcast(data, root=0)
for i in range(100):
    assert(data[i] == i)

# execute: mpiexec -n 2 python 06_broadcasting_np_array.py
# output: no output