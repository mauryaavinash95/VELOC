#include "client.hpp"
#include "include/veloc.h"
#include "common/file_util.hpp"
#include <vector>
#include <fstream>
#include <stdexcept>
#include <regex>
#include <unistd.h>
#include <limits.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#define __DEBUG
#include "common/debug.hpp"

static bool validate_name(const char *name) {
    std::regex e("[a-zA-Z0-9_\\.]+");
    return std::regex_match(name, e);
}

static void launch_backend(const char *cfg_file) {
    char *path = getenv("VELOC_BIN");
    std::string command;
    if (path != NULL)
        command = std::string(path) + "/";
    command += "veloc-backend " + std::string(cfg_file) + " --disable-ec > /dev/null";
    if (system(command.c_str()) != 0)
        FATAL("cannot launch active backend for async mode, error: " << strerror(errno));
}

veloc_client_t::veloc_client_t(unsigned int id, const char *cfg_file) :
    cfg(cfg_file), collective(false), rank(id) {
    if (cfg.is_sync()) {
	modules = new module_manager_t();
	modules->add_default_modules(cfg);
    } else {
        launch_backend(cfg_file);
	queue = new client_t<command_t>(rank);
    }
    ec_active = run_blocking(command_t(rank, command_t::INIT, 0, "")) > 0;
    cudaStreamCreate(&veloc_stream);
    gpu_memcpy_thread = std::thread([&] { gpu_to_host_trf(); });
    write_to_file_thread = std::thread([&] { mem_to_file_write(); });
    gpu_memcpy_thread.detach();
    write_to_file_thread.detach();
    DBG("VELOC initialized");
}

veloc_client_t::veloc_client_t(MPI_Comm c, const char *cfg_file) :
    cfg(cfg_file), comm(c), collective(true) {
    MPI_Comm_rank(comm, &rank);
    if (cfg.is_sync()) {
	modules = new module_manager_t();
	modules->add_default_modules(cfg, comm, true);
    } else {
        launch_backend(cfg_file);
	queue = new client_t<command_t>(rank);
    }
    ec_active = run_blocking(command_t(rank, command_t::INIT, 0, "")) > 0;
    cudaStreamCreate(&veloc_stream);
    gpu_memcpy_thread = std::thread([&] { gpu_to_host_trf(); });
    write_to_file_thread = std::thread([&] { mem_to_file_write(); });
    gpu_memcpy_thread.detach();
    write_to_file_thread.detach();
    DBG("VELOC initialized");
}

veloc_client_t::~veloc_client_t() {
    delete queue;
    delete modules;
    veloc_client_active = false;
    cudaStreamDestroy(veloc_stream);
    DBG("VELOC finalized");
}

bool veloc_client_t::mem_protect(int id, void *ptr, size_t count, size_t base_size, unsigned int flags=0, release release_routine=NULL ) {
    // mem_regions[id] = std::make_pair(ptr, base_size * count);
    mem_regions[id] = std::make_tuple(ptr, base_size * count, flags, release_routine);
    return true;
}

bool veloc_client_t::mem_unprotect(int id) {
    return mem_regions.erase(id) > 0;
}

bool veloc_client_t::checkpoint_wait() {
    if (cfg.is_sync())
	return true;
    if (checkpoint_in_progress) {
	ERROR("need to finalize local checkpoint first by calling checkpoint_end()");
	return false;
    }
    return queue->wait_completion() == VELOC_SUCCESS;
}

bool veloc_client_t::checkpoint_begin(const char *name, int version) {
    TIMER_START(io_timer_ckpt_begin);
    if (checkpoint_in_progress) {
	ERROR("nested checkpoints not yet supported");
	return false;
    }
    if (!validate_name(name) || version < 0) {
	ERROR("checkpoint name and/or version incorrect: name can only include [a-zA-Z0-9_] characters, version needs to be non-negative integer");
	return false;
    }

    DBG("called checkpoint_begin");
    current_ckpt = command_t(rank, command_t::CHECKPOINT, version, name);
    checkpoint_in_progress = true;

    file_stream.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    file_stream.open(current_ckpt.filename(cfg.get("scratch")), std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);

    ckpt_check_done = false;
    TIMER_STOP(io_timer_ckpt_begin, " --- CKPT BEGIN TIME --- ");
    return true;
}


void CUDART_CB veloc_client_t::enqueue_write(cudaStream_t stream, cudaError_t status, void *data) {
    veloc_client_t *th = (veloc_client_t *)data;
    int id = th->gpu_memcpy_queue.front();
    th->gpu_memcpy_queue.pop();
    std::pair<int, region_t> t = std::make_pair(id, th->ckpt_regions[id]);
    DBG("GPU ->Host Memcpy done for " << t.first << " now starting to write to file.");
    th->rem_gpu_cache += std::get<1>(t.second);
    release release_routine = std::get<3>(t.second); 
    // TODO: Call the release_routine function from here....
    std::unique_lock<std::mutex> write_queue_lock(th->write_to_file_mutex);
    th->write_to_file_regions.insert(t);
    th->write_to_file_cv.notify_one();
}

bool veloc_client_t::gpu_to_host_trf() {
    void *ptr; size_t sz;
    std::queue<std::pair<int, region_t>> local_gpu_memcpy_region_ids;
    while(veloc_client_active){ 
        do {
            std::unique_lock<std::mutex> lock(gpu_memcpy_mutex);
            while (gpu_memcpy_region_ids.empty()){
                gpu_memcpy_cv.wait(lock, [&](){ return (!gpu_memcpy_region_ids.empty() || ckpt_check_done); }); 
            }
            while(!gpu_memcpy_region_ids.empty()) {
                int id = gpu_memcpy_region_ids.front();
                local_gpu_memcpy_region_ids.push(std::make_pair(id, ckpt_regions[id]));
                gpu_memcpy_region_ids.pop();
            }
            lock.unlock();
            while (!local_gpu_memcpy_region_ids.empty()) {
                std::pair<int, region_t> e = local_gpu_memcpy_region_ids.front();
                local_gpu_memcpy_region_ids.pop();
                DBG("GPU memcpying region "<< e.first);
                char *temp;
                ptr = std::get<0>(e.second);
                sz = std::get<1>(e.second);
                cudaMallocHost((void**)&temp, sz);
                std::get<0>(ckpt_regions[e.first]) = temp;
                cudaMemcpyAsync(temp, ptr, sz, cudaMemcpyDeviceToHost, veloc_stream);
                gpu_memcpy_queue.push(e.first);
                cudaStreamAddCallback(veloc_stream, veloc_client_t::enqueue_write, this, 0);
                temp_host_ptrs.push_back(temp);
            }
        } while (!ckpt_check_done);
        cudaStreamSynchronize(veloc_stream);
    }
    return true;
}

bool veloc_client_t::checkpoint_mem(int mode, std::set<int> &ids) {
    TIMER_START(io_timer_ckpt_mem);
    DBG("Starting checkpoint_mem");
    if (!checkpoint_in_progress) {
        ERROR("must call checkpoint_begin() first");
        return false;
    }
    ckpt_regions.clear();
    if (mode == VELOC_CKPT_ALL)
        ckpt_regions = mem_regions;
    else if (mode == VELOC_CKPT_SOME) {
        for (auto it = ids.begin(); it != ids.end(); it++) {
            auto found = mem_regions.find(*it);
            if (found != mem_regions.end())
                ckpt_regions.insert(*found);
        }
    } else if (mode == VELOC_CKPT_REST) {
        ckpt_regions = mem_regions;
        for (auto it = ids.begin(); it != ids.end(); it++)
            ckpt_regions.erase(*it);
    }
    if (ckpt_regions.size() == 0) {
        ERROR("empty selection of memory regions to checkpoint, please check protection and/or selective checkpointing primitives");
        return false;
    }

    float host_cache = std::stof(current_ckpt.filename(cfg.get("host_cache_size")));
    float gpu_cache = std::stof(current_ckpt.filename(cfg.get("gpu_cache_size")));
    rem_gpu_cache = (1<<30)*gpu_cache;
    rem_host_cache = (1<<30)*host_cache;

    cudaPointerAttributes attributes;
    void *ptr; size_t sz;    
    unsigned int flags=0;

    for (auto &e : ckpt_regions) {
        sz = std::get<1>(e.second);
        if (sz > rem_host_cache) {
            ERROR("Please increase the host size to checkpoint region " << e.first);
            return false;
        }
    }

    for (auto &e : ckpt_regions) {
        ptr = std::get<0>(e.second);
        sz = std::get<1>(e.second);
        flags = std::get<2>(e.second);
        cudaPointerGetAttributes (&attributes, ptr);
        if(attributes.type==cudaMemoryTypeDevice) {
            if(rem_gpu_cache >= sz) {
                char *new_ptr = (char *)ptr;
                if(flags == DEFAULT) {
                    cudaMalloc((void**)&new_ptr, sz);
                    rem_gpu_cache -= sz;
                    DBG("Creating a copy on GPU cache for region " << e.first);
                    cudaMemcpy(new_ptr, ptr, sz, cudaMemcpyDeviceToDevice);
                    temp_dev_ptrs.push_back(new_ptr);
                }
                std::get<0>(e.second) = new_ptr;
                std::unique_lock<std::mutex> lock(gpu_memcpy_mutex);
                gpu_memcpy_region_ids.push(e.first);
                gpu_memcpy_cv.notify_one();
            } else {
                DBG("Not enough free cache on GPU for region " << e.first);
                char *temp;
                cudaMallocHost((void**)&temp, sz);
                cudaMemcpy(temp, ptr, sz, cudaMemcpyDeviceToHost);
                std::get<0>(e.second) = temp; 
                temp_host_ptrs.push_back(temp);  
                std::unique_lock<std::mutex> write_queue_lock(write_to_file_mutex);
                write_to_file_regions.insert(e);
                write_to_file_cv.notify_one();
            }
        } else {
            DBG("Direct host to file transfer for region " << e.first);
            std::unique_lock<std::mutex> write_queue_lock(write_to_file_mutex);
            write_to_file_regions.insert(e);
            write_to_file_cv.notify_one();
        }
    }
    ckpt_check_done = true;
    // Notify gpu_to_host_trf and mem_to_file_write CVs
    // that all checkpoint regions have been checked, i.e. parsed and sent to 
    // relevant threads for further execution.
    std::unique_lock<std::mutex> lock(gpu_memcpy_mutex);
    gpu_memcpy_cv.notify_one();
    std::unique_lock<std::mutex> write_queue_lock(write_to_file_mutex);
    write_to_file_cv.notify_one();

    TIMER_STOP(io_timer_ckpt_mem, " --- CKPT MEM TIME --- ");
    return true;
}

bool veloc_client_t::mem_to_file_write() {
    std::queue<std::pair<int, region_t>> local_write_regions; 
    bool is_writing = false;   
    while(veloc_client_active) {
        try {        
            do {
                std::unique_lock<std::mutex> lock(write_to_file_mutex);
                while (write_to_file_regions.empty()){
                    write_to_file_cv.wait(lock, [&](){ return (!write_to_file_regions.empty() || ckpt_check_done); }); 
                }
                while (!write_to_file_regions.empty()) {
                    is_writing = true;
                    auto e = *write_to_file_regions.begin();
                    local_write_regions.push(e);
                    write_to_file_regions.erase(e.first);
                }
                while (!local_write_regions.empty()) {
                    auto e = local_write_regions.front();
                    local_write_regions.pop();
                    int d = std::distance(ckpt_regions.begin(), ckpt_regions.find(e.first));
                    int offset = d + sizeof(size_t) + ckpt_regions.size()*(sizeof(size_t)+sizeof(int));
                    file_stream.seekp(offset);
                    DBG("Starting to write region " << e.first << " at offset: " << offset << " for " << (std::get<0>(e.second)) << " sz: " << std::get<1>(e.second));
                    file_stream.write((char *)&(std::get<0>(e.second)), std::get<1>(e.second));
                    DBG("f.write completed...");
                }
            } while (!ckpt_check_done || !write_to_file_regions.empty());
            if(is_writing) {
                // Write headers
                file_stream.seekp(0);
                size_t regions_size = ckpt_regions.size();
                file_stream.write((char *)&regions_size, sizeof(size_t));
                for (auto &e : ckpt_regions) {
                    file_stream.write((char *)&(e.first), sizeof(int));
                    file_stream.write((char *)&(std::get<1>(e.second)), sizeof(size_t));
                }  
                file_stream.close();
                DBG("DONE writing all regions to scratch!");
                for (char *t : temp_host_ptrs)
                    cudaFreeHost(t);
                for (char *t : temp_dev_ptrs)
                    cudaFree(t);
                temp_host_ptrs.clear();
                temp_dev_ptrs.clear();
                is_writing = false;
            }
        } catch (std::ofstream::failure &f) {
            ERROR("cannot write to checkpoint file: " << current_ckpt << ", reason: " << f.what());
            for (char *t : temp_host_ptrs)
                cudaFreeHost(t);
            for (char *t : temp_dev_ptrs)
                cudaFree(t);
            temp_host_ptrs.clear();
            temp_dev_ptrs.clear();
            return false;
        }
    }
    return true;
}

bool veloc_client_t::checkpoint_end(bool /*success*/) {
    TIMER_START(io_timer_ckpt_end);
    checkpoint_in_progress = false;
    if (cfg.is_sync()) {
        TIMER_STOP(io_timer_ckpt_end, " --- CKPT END TIME --- ");
        return modules->notify_command(current_ckpt) == VELOC_SUCCESS;
    }
    else {
        queue->enqueue(current_ckpt);
        TIMER_STOP(io_timer_ckpt_end, " --- CKPT END TIME --- ");
        return true;
    }
}

int veloc_client_t::run_blocking(const command_t &cmd) {
    if (cfg.is_sync())
	return modules->notify_command(cmd);
    else {
	queue->enqueue(cmd);
	return queue->wait_completion();
    }
}

int veloc_client_t::restart_test(const char *name, int needed_version) {
    if (!validate_name(name) || needed_version < 0) {
	ERROR("checkpoint name and/or version incorrect: name can only include [a-zA-Z0-9_] characters, version needs to be non-negative integer");
	return VELOC_FAILURE;
    }
    int version = run_blocking(command_t(rank, command_t::TEST, needed_version, name));
    DBG(name << ": latest version = " << version);
    if (collective) {
	int min_version;
	MPI_Allreduce(&version, &min_version, 1, MPI_INT, MPI_MIN, comm);
	return min_version;
    } else
	return version;
}

std::string veloc_client_t::route_file(const char *original) {
    char abs_path[PATH_MAX + 1];
    if (original[0] != '/' && getcwd(abs_path, PATH_MAX) != NULL)
	current_ckpt.assign_path(current_ckpt.original, std::string(abs_path) + "/" + std::string(original));
    else
	current_ckpt.assign_path(current_ckpt.original, std::string(original));
    return current_ckpt.filename(cfg.get("scratch"));
}

bool veloc_client_t::restart_begin(const char *name, int version) {
    if (checkpoint_in_progress) {
	INFO("cannot restart while checkpoint in progress");
	return false;
    }
    if (!validate_name(name) || version < 0) {
	ERROR("checkpoint name and/or version incorrect: name can only include [a-zA-Z0-9_] characters, version needs to be non-negative integer");
	return VELOC_FAILURE;
    }

    int result, end_result;
    current_ckpt = command_t(rank, command_t::RESTART, version, name);
    result = run_blocking(current_ckpt);
    if (collective)
	MPI_Allreduce(&result, &end_result, 1, MPI_INT, MPI_LOR, comm);
    else
	end_result = result;
    if (end_result == VELOC_SUCCESS) {
        header_size = 0;
	return true;
    } else
	return false;
}

bool veloc_client_t::read_header() {
    region_info.clear();
    try {
	std::ifstream f;
        size_t expected_size = 0;

	f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	f.open(current_ckpt.filename(cfg.get("scratch")), std::ifstream::in | std::ifstream::binary);
	size_t no_regions, region_size;
	int id;
	f.read((char *)&no_regions, sizeof(size_t));
	for (unsigned int i = 0; i < no_regions; i++) {
	    f.read((char *)&id, sizeof(int));
	    f.read((char *)&region_size, sizeof(size_t));
	    region_info.insert(std::make_pair(id, region_size));
            expected_size += region_size;
	}
	header_size = f.tellg();
        f.seekg(0, f.end);
        size_t file_size = (size_t)f.tellg() - header_size;
        if (file_size != expected_size)
            throw std::ifstream::failure("file size " + std::to_string(file_size) + " does not match expected size " + std::to_string(expected_size));
    } catch (std::ifstream::failure &e) {
	ERROR("cannot validate header for checkpoint " << current_ckpt << ", reason: " << e.what());
	header_size = 0;
	return false;
    }
    return true;
}

size_t veloc_client_t::recover_size(int id) {
    if (header_size == 0)
        read_header();
    auto it = region_info.find(id);
    if (it == region_info.end())
	return 0;
    else
	return it->second;
}

bool veloc_client_t::recover_mem(int mode, std::set<int> &ids) {
    if (header_size == 0 && !read_header()) {
	ERROR("cannot recover in memory mode if header unavailable or corrupted");
	return false;
    }
    try {
	std::ifstream f;
	f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	f.open(current_ckpt.filename(cfg.get("scratch")), std::ifstream::in | std::ifstream::binary);
	f.seekg(header_size);
	for (auto &e : region_info) {
	    bool found = ids.find(e.first) != ids.end();
	    if ((mode == VELOC_RECOVER_SOME && !found) || (mode == VELOC_RECOVER_REST && found)) {
		f.seekg(e.second, std::ifstream::cur);
		continue;
	    }
	    if (mem_regions.find(e.first) == mem_regions.end()) {
		ERROR("no protected memory region defined for id " << e.first);
		return false;
	    }
	    if (std::get<1>(mem_regions[e.first]) < e.second) {
		ERROR("protected memory region " << e.first << " is too small ("
		      << std::get<1>(mem_regions[e.first]) << ") to hold required size ("
		      << e.second << ")");
		return false;
	    }
	    f.read((char *)std::get<0>(mem_regions[e.first]), e.second);
	}
    } catch (std::ifstream::failure &e) {
	ERROR("cannot read checkpoint file " << current_ckpt << ", reason: " << e.what());
	return false;
    }
    return true;
}

bool veloc_client_t::restart_end(bool /*success*/) {
    return true;
}
