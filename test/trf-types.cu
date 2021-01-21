#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "veloc.h"
#include <assert.h>
using namespace std;
#define clock std::chrono::steady_clock
#define MB(x)   ((size_t) (x) >> 20)

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


__global__ void init_array(int *a, int n, int val=0) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    // Make sure we do not go out of bounds
    while (id < n) {
      a[id] = val;
      id += blockDim.x;
    }
}


int main(int argc, char *argv[]) {
    int size = 1<<20;
    int rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (VELOC_Init(MPI_COMM_WORLD, argv[1]) != VELOC_SUCCESS)
    {
        printf("Error initializing VELOC! Aborting...\n");
        exit(2);
    }
    cout << rank << " / " << world_size << " -> Starting for: " << MB(size) << " * 10^6 elements. " << endl; 
    int *ha;
    int *da;
    int *ma;
    double t;
    int ids[1];
    checkCuda(cudaMallocHost((void**)&ha, size*sizeof(int)));

    cudaMalloc((void**)&da, size*sizeof(int));
    VELOC_Mem_protect(0, da, size, sizeof(int));
    init_array<<<512, 512>>>(da, size, 0);
    checkCuda(cudaDeviceSynchronize());
    printf("Starting combined transfer.... unblocked\n");
    t = MPI_Wtime();
    assert(VELOC_Checkpoint("combined", 0) == VELOC_SUCCESS);
    assert(VELOC_Checkpoint_wait() == VELOC_SUCCESS);
    printf("Combined transfer completed in %lf s\n", MPI_Wtime()-t);

    cudaFree(da);
    checkCuda(cudaMalloc((void**)&da, size*sizeof(int)));  
    init_array<<<512, 512>>>(da, size, 0);
    checkCuda(cudaDeviceSynchronize());
    VELOC_Mem_protect(1, ha, size, sizeof(int));
    t = MPI_Wtime();
    printf("Starting transfer from device to host...\n");
    checkCuda(cudaMemcpy(ha, da, size*sizeof(int), cudaMemcpyDeviceToHost));
    printf("Device to Host transfer took %lf s\n", MPI_Wtime()-t);
    t = MPI_Wtime();
    VELOC_Checkpoint_begin("only_host", 0);
    ids[0] = 1;
    VELOC_Checkpoint_selective(VELOC_CKPT_SOME, ids, 1);
    VELOC_Checkpoint_end(1);
    assert(VELOC_Checkpoint_wait() == VELOC_SUCCESS);
    printf("VELOC memory based completed in %lf s\n", MPI_Wtime()-t);    

    memset(ha, 0, size*sizeof(int));
    checkCuda(cudaMallocManaged((void**)&ma, size*sizeof(int)));
    printf("Pinned %x, On-device %x, Unpinned %x, Managed %x \n", ha, da, &t, ma);
    init_array<<<512, 512>>>(ma, size, 0);
    checkCuda(cudaDeviceSynchronize());
    memcpy(ha, ma, size*sizeof(int));
    printf("Memcpy of managed memory is done: %x, %d\n", ha, ha[0]);

    VELOC_Mem_protect(2, ma, size, sizeof(int));
    t = MPI_Wtime();
    assert(VELOC_Checkpoint("managed_memory", 0) == VELOC_SUCCESS);
    assert(VELOC_Checkpoint_wait() == VELOC_SUCCESS);
    printf("Managed_memory transfer completed in %lf s\n", MPI_Wtime()-t);
    // cudaPointerAttributes attributes;
    // int *ha, *hb, *hc, *da, *db, *dc;
    // unsigned long long int s = 0, ds = 0;
    // int *unpinned, *managed;
    // long long int *host_var;
    // cudaMallocHost((void**)&host_var, sizeof(long long int));
    
    // unpinned = (int *)malloc(sizeof(int)*size);
    // cudaMallocManaged(&managed, sizeof(int)*size);
    // cudaMallocHost((void**)&ha, size*sizeof(int)); 
    // cudaMallocHost((void**)&hb, size*sizeof(int)); 
    // cudaMallocHost((void**)&hc, size*sizeof(int)); 
    // cudaMalloc((void**)&da, size*sizeof(int));
    // cudaMalloc((void**)&db, size*sizeof(int));
    // cudaMalloc((void**)&dc, size*sizeof(int));
    // // VELOC_Mem_protect(0, hc, size, sizeof(int));
    // s = fill_arrays(ha, hb, size);
    // cudaPointerGetAttributes (&attributes,unpinned);
    // printf("Memory type for unpinned memory array is %i\n",attributes.type);
    // cudaPointerGetAttributes (&attributes,ha);
    // printf("Memory type for pinned memory array is %i\n",attributes.type);
    // cudaPointerGetAttributes (&attributes,da);
    // printf("Memory type for on-device memory array is %i\n",attributes.type);
    // cudaPointerGetAttributes (&attributes,managed);
    // printf("Memory type for managed memory array is %i\n",attributes.type);

    // cout << rank << " / " << world_size << " -> Starting memory transfer..." << endl;
    // cudaMemcpyAsync(da, ha, size*sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpyAsync(db, hb, size*sizeof(int), cudaMemcpyHostToDevice);
    // kernel_dummy<<<512, 512>>>(da, db, dc, size);
    // cudaMemcpyAsync(hc, dc, size*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // double t = MPI_Wtime();
    // checkCuda(cudaMemcpyFromSymbol(host_var, global_var, sizeof(long long int), 0, cudaMemcpyDeviceToHost));
    // printf("Time taken for memcpyfromsymbol is: %lf, value is: %l\n", MPI_Wtime()-t, *host_var);
    // // assert(VELOC_Checkpoint("sample_app", 1) == VELOC_SUCCESS);
    // // assert(VELOC_GPU_Async_Checkpoint("sample_app", 1, 0, da, ha, size, sizeof(int)) == VELOC_SUCCESS);
    // cout << rank << " / " << world_size << " -> Kernel computation completed..." << endl;
    // for(int i=0; i<size; i++) {
    //     ds += hc[i];
    // }
    // if (ds != s) {
    //     cout << rank << " / " << world_size << " -> The kernel_dummy did not compute the sums properly. Expected: " 
    //       << s << ", got: " << ds << endl;
    // } else {
    //     cout << rank << " / " << world_size << " -> Sums match!" << endl;
    // }
    // cudaFreeHost(ha);
    // cudaFreeHost(hb);
    // cudaFreeHost(hc);
    // cudaFree(da);
    // cudaFree(db);
    // cudaFree(dc);
    VELOC_Finalize(0);
    MPI_Finalize();
    return 0;
}
