#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#else
#include <sched.h>
#include <unistd.h>
#endif

// pin the current thread to a core (0-based)
inline void pin_thread_to_core(int core_id) {
#ifdef __APPLE__
    thread_affinity_policy_data_t policy = {core_id};
    thread_port_t thread = mach_thread_self();
    thread_policy_set(thread, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy,
                      THREAD_AFFINITY_POLICY_COUNT);
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

// raise thread priority (best effort)
inline void set_high_priority() {
#ifdef __APPLE__
    pthread_t t = pthread_self();
    sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(t, SCHED_FIFO, &param);
#else
    // Linux: nice value -20 is highest
    nice(-20);
#endif
}

std::vector<Ray> load_rays_binary(const std::string &filename,
                                  int64_t ray_count) {
    std::vector<Ray> rays;
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Error: Could not open file " << filename
                  << " for reading\n";
        return rays;
    }

    // Read number of rays
    size_t count;
    file.read(reinterpret_cast<char *>(&count), sizeof(count));
    if (ray_count > count) {
        std::cout << "the requested ray count: " << ray_count
                  << " is greater than the total ray count: " << count
                  << " You need to re-generate the rays." << std::endl;
    }
    assert(ray_count <= count);

    rays.reserve(ray_count);

    // Read ray data
    for (size_t i = 0; i < ray_count; ++i) {
        Ray ray;
        file.read(reinterpret_cast<char *>(&ray.o[0]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.o[1]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.o[2]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d[0]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d[1]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d[2]), sizeof(float));
        rays.push_back(ray);
    }

    file.close();
    return rays;
}
