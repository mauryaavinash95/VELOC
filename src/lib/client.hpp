#ifndef __CLIENT_HPP
#define __CLIENT_HPP

#include "common/config.hpp"
#include "common/command.hpp"
#include "common/comm_queue.hpp"
#include "modules/module_manager.hpp"
#include "include/veloc.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <unordered_map>
#include <map>
#include <set>
#include <deque>
#include <queue> 
#include <string>
#include <fstream>

class veloc_client_t {
    config_t cfg;
    MPI_Comm comm;
    bool collective, ec_active;
    int rank;

    // typedef std::pair <void *, size_t> region_t;
    typedef std::tuple <void *, size_t, unsigned int, release> region_t;
    typedef std::map<int, region_t> regions_t;

    regions_t mem_regions;
    regions_t ckpt_regions;
    command_t current_ckpt;
    cudaStream_t veloc_stream;
    bool checkpoint_in_progress = false;
    bool ckpt_check_done = true;

    // regions_t gpu_memcpy_regions;
    std::queue<int> gpu_memcpy_region_ids;
    std::queue<int> gpu_memcpy_queue;
    std::vector<void *> gpu_memcpy_new_ptrs;
    std::mutex gpu_memcpy_mutex;
    std::condition_variable gpu_memcpy_cv;
    std::thread gpu_memcpy_thread;
    
    float rem_gpu_cache = 0;
    float rem_host_cache = 0;
    regions_t write_to_file_regions;
    std::mutex write_to_file_mutex;
    std::condition_variable write_to_file_cv;
    std::thread write_to_file_thread;

    bool gpu_memcpy_done = true;
    std::mutex gpu_memcpy_done_mutex;
    std::condition_variable gpu_memcpy_done_cv; 

    bool veloc_client_active = true;
    std::ofstream file_stream;  

    std::vector<char *> temp_host_ptrs;
    std::vector<char *> temp_dev_ptrs;

    std::map<int, size_t> region_info;
    size_t header_size = 0;

    client_t<command_t> *queue = NULL;
    module_manager_t *modules = NULL;

    int run_blocking(const command_t &cmd);
    bool read_header();

public:
    veloc_client_t(unsigned int id, const char *cfg_file);
    veloc_client_t(MPI_Comm comm, const char *cfg_file);

    bool mem_protect(int id, void *ptr, size_t count, size_t base_size, unsigned int flags, release release_routine);
    bool mem_unprotect(int id);
    std::string route_file(const char *original);

    bool checkpoint_begin(const char *name, int version);
    bool checkpoint_mem(int mode, std::set<int> &ids);
    bool checkpoint_end(bool success);
    bool checkpoint_wait();
    static void CUDART_CB enqueue_write(cudaStream_t stream, cudaError_t status, void *data); 
    bool gpu_to_host_trf();
    bool mem_to_file_write();
    bool write_headers(regions_t ckpt_regions);
    int restart_test(const char *name, int version);
    bool restart_begin(const char *name, int version);
    size_t recover_size(int id);
    bool recover_mem(int mode, std::set<int> &ids);
    bool restart_end(bool success);

    ~veloc_client_t();
};

#endif
