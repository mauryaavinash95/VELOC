#include "include/veloc.h"
#include "client.hpp"
// #include <cuda_runtime.h>
static veloc_client_t *veloc_client = NULL;

#define __DEBUG
#include "common/debug.hpp"

void __attribute__ ((constructor)) veloc_constructor() {
}

void __attribute__ ((destructor)) veloc_destructor() {
}

extern "C" int VELOC_Init(MPI_Comm comm, const char *cfg_file) {
    try {
	veloc_client = new veloc_client_t(comm, cfg_file);
	return VELOC_SUCCESS;
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
	return VELOC_FAILURE;
    }
}

extern "C" int VELOC_Init_single(unsigned int id, const char *cfg_file) {
    try {
	veloc_client = new veloc_client_t(id, cfg_file);
	return VELOC_SUCCESS;
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
	return VELOC_FAILURE;
    }
}

#define CLIENT_CALL(x) (veloc_client != NULL && (x)) ? VELOC_SUCCESS : VELOC_FAILURE;

extern "C" int VELOC_Mem_protect(int id, void *ptr, size_t count, size_t base_size) {
    return CLIENT_CALL(veloc_client->mem_protect(id, ptr, count, base_size));
}

extern "C" int VELOC_Mem_unprotect(int id) {
    return CLIENT_CALL(veloc_client->mem_unprotect(id));
}

extern "C" int VELOC_Checkpoint_begin(const char *name, int version) {
    return CLIENT_CALL(veloc_client->checkpoint_begin(name, version));
}

extern "C" int VELOC_Checkpoint_mem() {
    std::set<int> id_set = {};
    return CLIENT_CALL(veloc_client->checkpoint_mem(VELOC_CKPT_ALL, id_set));
}

extern "C" int VELOC_Checkpoint_selective(int mode, int *ids, int no_ids) {
    std::set<int> id_set = {};
    for (int i = 0; i < no_ids; i++)
	id_set.insert(ids[i]);
    return CLIENT_CALL(veloc_client->checkpoint_mem(mode, id_set));
}

extern "C" int VELOC_Checkpoint_end(int success) {
    return CLIENT_CALL(veloc_client->checkpoint_end(success));
}

extern "C" int VELOC_Checkpoint_wait() {
    return CLIENT_CALL(veloc_client->checkpoint_wait());
}

extern "C" int VELOC_Restart_test(const char *name, int version) {
    if (veloc_client == NULL)
	return -1;
    return veloc_client->restart_test(name, version);
}

extern "C" int VELOC_Route_file(const char *original, char *routed) {
    std::string cname = veloc_client->route_file(original);
    cname.copy(routed, cname.length());
    routed[cname.length()] = 0;

    return routed[0] != 0 ? VELOC_SUCCESS : VELOC_FAILURE;
}

extern "C" int VELOC_Restart_begin(const char *name, int version) {
    return CLIENT_CALL(veloc_client->restart_begin(name, version));
}

extern "C" int VELOC_Recover_mem() {
    std::set<int> id_set = {};
    return CLIENT_CALL(veloc_client->recover_mem(VELOC_RECOVER_ALL, id_set));
}

extern "C" int VELOC_Recover_selective(int mode, int *ids, int no_ids) {
    std::set<int> id_set = {};
    for (int i = 0; i < no_ids; i++)
	id_set.insert(ids[i]);
    return CLIENT_CALL(veloc_client->recover_mem(mode, id_set));
}

extern "C" int VELOC_Recover_size(int id) {
    return veloc_client->recover_size(id);
}

extern "C" int VELOC_Restart_end(int success) {
    return CLIENT_CALL(veloc_client->restart_end(success));
}

extern "C" int VELOC_Restart(const char *name, int version) {
    int ret = VELOC_Restart_begin(name, version);
    if (ret == VELOC_SUCCESS)
	ret = VELOC_Recover_mem();
    if (ret == VELOC_SUCCESS)
	ret = VELOC_Restart_end(1);
    return ret;
}

extern "C" int VELOC_Checkpoint(const char *name, int version) {
    int ret = VELOC_Checkpoint_wait();
    if (ret == VELOC_SUCCESS)
	ret = VELOC_Checkpoint_begin(name, version);
    if (ret == VELOC_SUCCESS)
	ret = VELOC_Checkpoint_mem();
    if (ret == VELOC_SUCCESS)
	ret = VELOC_Checkpoint_end(1);
    return ret;
}

// extern "C" int VELOC_GPU_Async_Checkpoint(const char *name, int version, int mem_prot_id, void *d_ptr, void *h_ptr, size_t count, size_t base_size) {
//     int ret = VELOC_Checkpoint_wait();
//     int ids[] = {mem_prot_id};
//     if (ret == VELOC_SUCCESS)
//     cudaMemcpy(h_ptr, d_ptr, count*base_size, cudaMemcpyDeviceToHost);
//     ret = VELOC_Mem_protect(mem_prot_id, h_ptr, count, base_size);
//     if (ret == VELOC_SUCCESS)
// 	ret = VELOC_Checkpoint_begin(name, version);
//     if (ret == VELOC_SUCCESS)
// 	ret = VELOC_Checkpoint_selective(VELOC_CKPT_SOME, ids, 1);
//     if (ret == VELOC_SUCCESS)
// 	ret = VELOC_Checkpoint_end(1);
//     return ret;
// }

extern "C" int VELOC_Finalize(int drain) {
    if (veloc_client != NULL) {
        int ret = VELOC_SUCCESS;
	if (drain)
	    ret = VELOC_Checkpoint_wait();
	delete veloc_client;
	return ret;
    } else {
	ERROR("Attempting to finalize VELOC before it was initialized");
	return VELOC_FAILURE;
    }
}
