#!makefile

MPI_PATH=/work/soft/mpi/openmpi/3.1.0_hygon/gcc-7.2
#MPI_PATH=/opt/soft/mpi/openmpi-2.1.2
test.out: main.cpp
	hipcc -I${MPI_PATH}/include -g -o $@ $< -L${MPI_PATH}/lib -lmpi -lmpi_cxx

clean:
	rm -rf *.o *.out




