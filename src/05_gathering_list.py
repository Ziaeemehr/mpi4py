from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Create a list of objects on each process
my_objects = [f"object_{rank}_{i}" for i in range(3)]

# Gather all lists onto the root process
if rank == 0:
    all_objects = []
else:
    all_objects = None

all_objects = comm.gather(my_objects, root=0)

# Print the results on the root process
if rank == 0:
    print("All objects:")
    for obj_list in all_objects:
        for obj in obj_list:
            print(obj)
