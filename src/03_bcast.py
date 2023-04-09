# Collective Communication
# Broadcast
# Broadcasts a message from the process with rank "root" 
# to all other processes of the communicator.

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {'key1': [7, 2.72, 2+3j],
            'key2': ( 'abc', 'xyz')}
else:
    data = None
data = comm.bcast(data, root=0)
print(rank, data)

# execute: mpiexec -n 4 python 03_bcast.py
# output: 
# 1 {'key1': [7, 2.72, (2+3j)], 'key2': ('abc', 'xyz')}
# 0 {'key1': [7, 2.72, (2+3j)], 'key2': ('abc', 'xyz')}