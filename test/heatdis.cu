#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "veloc.h"
#include "heatdis.h"
#define NUM_BLOCKS 512
#define NUM_THREADS 512
static const unsigned int CKPT_FREQ = ITER_TIMES / 3;
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

/*
    This sample application is based on the heat distribution code
    originally developed within the FTI project: github.com/leobago/fti
*/
__device__ long long int error_val;
__global__ void init_data_gpu(int nbLines, int M, int rank, double *h) {
    int j;
    error_val = 0;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    while (idx < nbLines) {
        for (j = 0; j < M; j++) {
            h[(idx*M)+j] = 0;
        }
        idx += blockDim.x;
    }
    if (rank == 0) {
        int start = int(M*0.1), end = ceil(M*0.9);
        idx = threadIdx.x + blockIdx.x*blockDim.x;
        while(idx >= start && idx < end) {
            h[idx] = 100;
            idx += blockDim.x;
        }
    }
}

__global__ void copy_g(int nbLines, int M, double *g, double *h) {
    int j, idx = threadIdx.x + blockIdx.x*blockDim.x;
    while (idx < nbLines) {
        for(j = 0; j < M; j++) {
            h[(idx*M)+j] = g[(idx*M)+j];
        }
        idx += blockDim.x;
    }
}

__global__ void compute(int nbLines, int M, double *g, double *h) {
    int j, i = threadIdx.x + blockIdx.x*blockDim.x;
    double err;
    __shared__ long long int block_err;
    block_err = 0;
    long long int le;
    while (i > 0 && i < (nbLines-1)) {
        for(j = 0; j < M; j++) {
            g[(i*M)+j] = 0.25*(h[((i-1)*M)+j]+h[((i+1)*M)+j]+h[(i*M)+j-1]+h[(i*M)+j+1]);
            err = fabs(g[(i*M)+j] - h[(i*M)+j]);
            le = __double_as_longlong(err);
            // atomicMax(&error_val, le);
            atomicMax(&block_err, le);
        }
        i += blockDim.x;
    }
    atomicMax(&error_val, block_err);
}

__global__ void compute_right(int nbLines, int M, double *g) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    while(idx < M) {
        g[((nbLines-1)*M)+idx] = g[((nbLines-2)*M)+idx];
        idx += blockDim.x;
    }
}

double doWork(int numprocs, int rank, int M, int nbLines, double *dg, double *dh) {
    MPI_Request req1[2], req2[2];
    MPI_Status status1[2], status2[2];
    long long int *localerror;
    checkCuda(cudaMallocHost((void**)&localerror, sizeof(long long int)));
    checkCuda(cudaMemcpyToSymbol(error_val, &localerror, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    double *g, *h;
    checkCuda(cudaMallocHost((void**)&h, sizeof(double *) * M * nbLines));
    checkCuda(cudaMallocHost((void**)&g, sizeof(double *) * M * nbLines));
    copy_g<<<NUM_BLOCKS, NUM_THREADS>>>(nbLines, M, dg, dh);
    cudaMemcpy(h, dh, sizeof(double *) * M * nbLines, cudaMemcpyDeviceToHost);
    cudaMemcpy(g, dg, sizeof(double *) * M * nbLines, cudaMemcpyDeviceToHost);
    if (rank > 0) {
        MPI_Isend(g+M, M, MPI_DOUBLE, rank-1, WORKTAG, MPI_COMM_WORLD, &req1[0]);
        MPI_Irecv(h,   M, MPI_DOUBLE, rank-1, WORKTAG, MPI_COMM_WORLD, &req1[1]);
    }
    if (rank < numprocs - 1) {
        MPI_Isend(g+((nbLines-2)*M), M, MPI_DOUBLE, rank+1, WORKTAG, MPI_COMM_WORLD, &req2[0]);
        MPI_Irecv(h+((nbLines-1)*M), M, MPI_DOUBLE, rank+1, WORKTAG, MPI_COMM_WORLD, &req2[1]);
    }
    if (rank > 0) {
        MPI_Waitall(2,req1,status1);
    }
    if (rank < numprocs - 1) {
        MPI_Waitall(2,req2,status2);
    }
    compute<<<NUM_BLOCKS, NUM_THREADS>>>(nbLines, M, dg, dh);
    if (rank == (numprocs-1)) {
        compute_right<<<NUM_BLOCKS, NUM_THREADS>>>(nbLines, M, dg);
    }
    cudaDeviceSynchronize();
    double t = MPI_Wtime();
    checkCuda(cudaMemcpyFromSymbol(localerror, error_val, sizeof(long long int), 0, cudaMemcpyDeviceToHost));
    // printf("Time taken for MemcpyFromSymbol is: %lf\n", MPI_Wtime()-t);
    cudaFreeHost(h);
    cudaFreeHost(g);
    return *((double*)localerror);
}

int main(int argc, char *argv[]) {
    int rank, nbProcs, nbLines, i, M, arg;
    double wtime, memSize, localerror, globalerror = 1, inner_time;
    if (argc < 3) {
        printf("Usage: %s <mem_in_mb> <cfg_file>\n", argv[0]);
        exit(1);
    }
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nbProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int cudaDevices = 0;
    checkCuda(cudaGetDeviceCount(&cudaDevices));
    checkCuda(cudaSetDevice(rank%cudaDevices));
    if (sscanf(argv[1], "%d", &arg) != 1) {
        printf("Wrong memory size! See usage\n");
	    exit(3);
    }
    if (VELOC_Init(MPI_COMM_WORLD, argv[2]) != VELOC_SUCCESS) {
        printf("Error initializing VELOC! Aborting...\n");
        exit(2);
    }
	
    M = (int)sqrt((double)(arg * 1024.0 * 1024.0 * nbProcs) / (2 * sizeof(double))); // two matrices needed
    nbLines = (M / nbProcs) + 3;
    
    double *dh, *dg;
    checkCuda(cudaMalloc((void**)&dh, sizeof(double *) * M * nbLines));
    checkCuda(cudaMalloc((void**)&dg, sizeof(double *) * M * nbLines));
    init_data_gpu<<<NUM_BLOCKS, NUM_THREADS>>>(nbLines, M, rank, dg);
    memSize = M * nbLines * 2 * sizeof(double) / (1024 * 1024);

    if (rank == 0)
	    printf("Local data size is %d x %d = %f MB (%d).\n", M, nbLines, memSize, arg);
    if (rank == 0)
	    printf("Target precision : %f \n", PRECISION);
    if (rank == 0)
	    printf("Maximum number of iterations : %d \n", ITER_TIMES);
    wtime = MPI_Wtime();
    inner_time = MPI_Wtime();
    i = 0;
    VELOC_Mem_protect(0, &i, 1, sizeof(int));
    VELOC_Mem_protect(1, dh, M * nbLines, sizeof(double));
    VELOC_Mem_protect(2, dg, M * nbLines, sizeof(double));
    while(i < ITER_TIMES) {
        localerror = doWork(nbProcs, rank, M, nbLines, dg, dh);
        if (((i % ITER_OUT) == 0) && (rank == 0)) {
            printf("Step : %d, error = %f, localerror = %f, time = %lf s\n", i, globalerror, localerror, MPI_Wtime()-inner_time);
            inner_time = MPI_Wtime();
        }
        if ((i % REDUCE) == 0) {
            MPI_Allreduce(&localerror, &globalerror, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        }
        if (globalerror < PRECISION) {
            printf("Breaking due to error being less: %lf", globalerror);
            break;
        }
        i++;
        if (i % CKPT_FREQ == 0) {
            printf("\n Checkpointing %i\n", i);
            assert(VELOC_Checkpoint("heatdis", i) == VELOC_SUCCESS);
        }
    }
    if (rank == 0)
	    printf("Execution finished in %lf seconds.\n", MPI_Wtime() - wtime);

    cudaFree(dh);
    cudaFree(dg);
    MPI_Finalize();
    return 0;
}
