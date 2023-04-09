# Sending a tensor from rank 0 to all other ranks
# first convert the tensor to a numpy array
# then use the numpy array as the send buffer
# then convert the received numpy array to a tensor


from mpi4py import MPI
import numpy as np
import torch

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sendbuf = None
if rank == 0:
    sendbuf = torch.empty([size,5], dtype=torch.float32)
    sendbuf.T[:,:] = torch.arange(size, dtype=torch.float32) 
    sendbuf = sendbuf.numpy()

recvbuf = np.empty(5, dtype=np.float32)
comm.Scatter(sendbuf, recvbuf, root=0)
recvbuf = torch.from_numpy(recvbuf)
assert(torch.allclose(recvbuf, torch.tensor(rank, dtype=torch.float32)))

