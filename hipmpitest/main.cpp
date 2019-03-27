#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include <mpi.h>
#include "fun.h"


void gpu_main(void)
{
	for (int i = 0; i < 1000; i++) {
		do_compute();

		if (!(i % 100))
			printf("i=%d\n", i);
	}
	printf("done\n");
}

int main(int argc, char **argv)
{
	int pid=-1, np=-1;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	MPI_Barrier(MPI_COMM_WORLD);

	if (!pid)
		gpu_main();

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();
	return 0;
}
