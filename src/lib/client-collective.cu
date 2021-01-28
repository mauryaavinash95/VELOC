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
    DBG("VELOC initialized");
}

veloc_client_t::~veloc_client_t() {
    delete queue;
    delete modules;
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
    TIMER_STOP(io_timer_ckpt_begin, " --- CKPT BEGIN TIME --- ");
    return true;
}

bool veloc_client_t::checkpoint_gpu_mem(regions_t async_gpu_regions) {
    void *ptr; size_t sz;
    for(auto &e: async_gpu_regions) {
        char *temp;
        ptr = std::get<0>(e.second);
        sz = std::get<1>(e.second);
        cudaMallocHost((void**)&temp, sz);
        cudaMemcpyAsync(temp, ptr, sz, cudaMemcpyDeviceToHost, veloc_stream);
        std::get<0>(async_gpu_regions[e.first]) = temp;
    }
    cudaStreamSynchronize(veloc_stream);
    return mem_write(async_gpu_regions);
}

bool veloc_client_t::checkpoint_mem(int mode, std::set<int> &ids) {
    TIMER_START(io_timer_ckpt_mem);
    DBG("Starting checkpoint_mem");
    if (!checkpoint_in_progress) {
        ERROR("must call checkpoint_begin() first");
        return false;
    }
    regions_t ckpt_regions;
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

    double gpu_cache = std::stod(current_ckpt.filename(cfg.get("gpu_cache_size")));
    double rem_gpu_cache = (1<<30)*gpu_cache;
    DBG("Allowed " << rem_gpu_cache << " GPU cache size.");
    regions_t async_gpu_regions;
    cudaPointerAttributes attributes;
    void *ptr; size_t sz; unsigned int flags=0; release release_routine=NULL;
    std::vector<char *> temp_ptrs;
    size_t free_gpu_mem, total_gpu_mem;
    for (auto &e : ckpt_regions) {
        ptr = std::get<0>(e.second);
        sz = std::get<1>(e.second);
        flags = std::get<2>(e.second);
        release_routine = std::get<3>(e.second); 
        cudaPointerGetAttributes (&attributes, ptr);
        if(attributes.type==cudaMemoryTypeDevice) {
            cudaMemGetInfo(&free_gpu_mem, &total_gpu_mem);
            char *new_ptr;
            if(free_gpu_mem >= sz && rem_gpu_cache >= sz) {
                if(flags == DEFAULT) {
                    cudaMalloc((void**)&new_ptr, sz);
                    cudaMemcpy(new_ptr, ptr, sz, cudaMemcpyDeviceToDevice);
                } else 
                    new_ptr = (char *)ptr;
                rem_gpu_cache -= sz;
                async_gpu_regions[e.first] = std::make_tuple(new_ptr, sz, flags, release_routine);
                continue;
            } else {
                char *temp;
                cudaMallocHost((void**)&temp, sz);
                cudaMemcpy(temp, new_ptr, sz, cudaMemcpyDeviceToHost);
                temp_ptrs.push_back(temp);
                ckpt_regions[e.first] = std::make_tuple(temp, sz, flags, release_routine);    
            }
        }
    }

    for (auto &e : async_gpu_regions)
        ckpt_regions.erase(e.first);

    TIMER_START(io_timer_ckpt_gpu_mem);
    gpu_memcpy_thread = std::thread([&] { mem_write(async_gpu_regions); });
    TIMER_START(io_timer_ckpt_host_mem);
    bool ret = mem_write(ckpt_regions);
    TIMER_STOP(io_timer_ckpt_host_mem, " --- CKPT HOST MEM TIME --- ");
    gpu_memcpy_thread.join();
    TIMER_STOP(io_timer_ckpt_gpu_mem, " --- CKPT GPU MEM TIME --- ");

    for (char *t : temp_ptrs)
        cudaFreeHost(t);

    TIMER_STOP(io_timer_ckpt_mem, " --- CKPT MEM TIME --- ");
    return ret;
}

bool veloc_client_t::mem_write(regions_t ckpt_regions) {
    std::ofstream f;
    f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    try {
        f.open(current_ckpt.filename(cfg.get("scratch")), std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
        size_t regions_size = ckpt_regions.size();
        f.write((char *)&regions_size, sizeof(size_t));
        for (auto &e : ckpt_regions) {
            f.write((char *)&(e.first), sizeof(int));
            f.write((char *)&(std::get<1>(e.second)), sizeof(size_t));
        }  

        for (auto &e : ckpt_regions)
            f.write((char *)std::get<0>(e.second), std::get<1>(e.second));
    } catch (std::ofstream::failure &f) {
        ERROR("cannot write to checkpoint file: " << current_ckpt << ", reason: " << f.what());
        return false;
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
