// What accelerators the phone has. Everything is opened with dlopen() at run time, so a phone
// without Vulkan/OpenCL just reports "not available"; this code runs in the probe process,
// because a broken GPU driver can crash inside these calls.

#include "mv_native.h"

#include <dlfcn.h>

#include <string>
#include <vector>

#if defined(__ANDROID__)
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <android/NeuralNetworks.h>
#endif

std::vector<std::string> mv_vulkan_devices() {
#if defined(__ANDROID__)
    void * lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        return {"error: no Vulkan driver"};
    }
    auto getProc = (PFN_vkGetInstanceProcAddr) dlsym(lib, "vkGetInstanceProcAddr");
    auto create = (PFN_vkCreateInstance) getProc(nullptr, "vkCreateInstance");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "myvoice";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (!create || create(&ci, nullptr, &inst) != VK_SUCCESS) {
        dlclose(lib);
        return {"error: Vulkan instance could not be created"};
    }
    auto enumerate = (PFN_vkEnumeratePhysicalDevices) getProc(inst, "vkEnumeratePhysicalDevices");
    auto props = (PFN_vkGetPhysicalDeviceProperties) getProc(inst, "vkGetPhysicalDeviceProperties");
    auto destroy = (PFN_vkDestroyInstance) getProc(inst, "vkDestroyInstance");
    uint32_t n = 0;
    enumerate(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    enumerate(inst, &n, devs.data());
    std::vector<std::string> out;
    for (auto d : devs) {
        VkPhysicalDeviceProperties p{};
        props(d, &p);
        const char * type = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "integrated GPU"
                          : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "GPU"
                          : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? "CPU" : "other";
        out.push_back(std::string(p.deviceName) + "\tVulkan " + std::to_string(VK_API_VERSION_MAJOR(p.apiVersion)) + "." +
                      std::to_string(VK_API_VERSION_MINOR(p.apiVersion)) + "\t" + std::to_string(p.driverVersion) + "\t" + type);
    }
    destroy(inst, nullptr);
    dlclose(lib);
    return out;
#else
    return {"error: not Android"};
#endif
}

std::vector<std::string> mv_opencl_devices() {
#if defined(__ANDROID__)
    void * lib = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        return {"error: no OpenCL driver (or not accessible to apps)"};
    }
    auto getPlatforms = (cl_int (*)(cl_uint, cl_platform_id *, cl_uint *)) dlsym(lib, "clGetPlatformIDs");
    auto platformInfo = (cl_int (*)(cl_platform_id, cl_platform_info, size_t, void *, size_t *)) dlsym(lib, "clGetPlatformInfo");
    auto getDevices = (cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *)) dlsym(lib, "clGetDeviceIDs");
    auto deviceInfo = (cl_int (*)(cl_device_id, cl_device_info, size_t, void *, size_t *)) dlsym(lib, "clGetDeviceInfo");
    if (!getPlatforms || !platformInfo || !getDevices || !deviceInfo) {
        dlclose(lib);
        return {"error: OpenCL library is incomplete"};
    }
    cl_uint np = 0;
    if (getPlatforms(0, nullptr, &np) != CL_SUCCESS || np == 0) {
        dlclose(lib);
        return {"error: no OpenCL platform"};
    }
    std::vector<cl_platform_id> ps(np);
    getPlatforms(np, ps.data(), nullptr);
    std::vector<std::string> out;
    for (auto p : ps) {
        char pname[256] = {0};
        platformInfo(p, CL_PLATFORM_NAME, sizeof(pname) - 1, pname, nullptr);
        cl_uint nd = 0;
        if (getDevices(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) {
            continue;
        }
        std::vector<cl_device_id> ds(nd);
        getDevices(p, CL_DEVICE_TYPE_GPU, nd, ds.data(), nullptr);
        for (auto d : ds) {
            char dname[256] = {0}, ver[256] = {0};
            deviceInfo(d, CL_DEVICE_NAME, sizeof(dname) - 1, dname, nullptr);
            deviceInfo(d, CL_DEVICE_VERSION, sizeof(ver) - 1, ver, nullptr);
            out.push_back(std::string(dname) + "\t" + ver + "\t" + pname);
        }
    }
    dlclose(lib);
    return out;
#else
    return {"error: not Android"};
#endif
}

std::vector<std::string> mv_nnapi_devices() {
#if defined(__ANDROID__)
    void * lib = dlopen("libneuralnetworks.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        return {"error: NNAPI not available"};
    }
    auto count = (int (*)(uint32_t *)) dlsym(lib, "ANeuralNetworks_getDeviceCount");
    auto get = (int (*)(uint32_t, ANeuralNetworksDevice **)) dlsym(lib, "ANeuralNetworks_getDevice");
    auto name = (int (*)(const ANeuralNetworksDevice *, const char **)) dlsym(lib, "ANeuralNetworksDevice_getName");
    auto type = (int (*)(const ANeuralNetworksDevice *, int32_t *)) dlsym(lib, "ANeuralNetworksDevice_getType");
    auto version = (int (*)(const ANeuralNetworksDevice *, const char **)) dlsym(lib, "ANeuralNetworksDevice_getVersion");
    if (!count || !get || !name || !type || !version) {
        dlclose(lib);
        return {"error: NNAPI device list not available"};
    }
    uint32_t n = 0;
    count(&n);
    std::vector<std::string> out;
    for (uint32_t i = 0; i < n; i++) {
        ANeuralNetworksDevice * d = nullptr;
        if (get(i, &d) != ANEURALNETWORKS_NO_ERROR) {
            continue;
        }
        const char * nm = "";
        const char * ver = "";
        int32_t t = 0;
        name(d, &nm);
        type(d, &t);
        version(d, &ver);
        const char * ts = t == ANEURALNETWORKS_DEVICE_ACCELERATOR ? "NPU/DSP"
                        : t == ANEURALNETWORKS_DEVICE_GPU ? "GPU"
                        : t == ANEURALNETWORKS_DEVICE_CPU ? "CPU" : "other";
        out.push_back(std::string(nm) + "\t" + ts + "\t" + ver);
    }
    dlclose(lib);
    return out;
#else
    return {"error: not Android"};
#endif
}
