# point to point communication
# send and receive data between two processes
# send data from process 0 to process 1
# receive data in process 1

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# print("Hello, World! I am process %d of %d." % (rank, comm.Get_size()))

if rank == 0:
    data = {'a': 7, 'b': 3.14}
    comm.send(data, dest=1, tag=11)
elif rank == 1:
    data = comm.recv(source=0, tag=11)
    print(data)

# execute: mpiexec -n 2 python 00_send_recieve.py
# output: {'a': 7, 'b': 3.14}
