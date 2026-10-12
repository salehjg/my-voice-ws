// Loading ggml's backend modules. The app ships every CPU variant ggml builds for Android
// (armv8.0, armv8.2 + dot product, + fp16, armv8.6 + int8 matmul, ...); each one reports a
// score for the CPU it runs on, and the best one is loaded. That is the phone's version of the
// laptop's "no AVX-512 here" problem: never run code the CPU can't execute.

#include "mv_native.h"

#include "ggml-backend.h"

#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>

#include <set>
#include <sstream>

static bool g_cpu_loaded = false;
static std::set<std::string> g_loaded;

static bool file_exists(const std::string & p) {
    struct stat st{};
    return stat(p.c_str(), &st) == 0;
}

#ifdef GGML_BACKEND_DL
// ggml dlclose()s a backend that finds no device, but the library may already have registered
// thread-local destructors; a thread that exits later then jumps into unmapped code (seen with
// the Vulkan backend on the Android emulator). Keep every backend we load mapped for good.
static void pin(const std::string & path) {
    dlopen(path.c_str(), RTLD_NOW | RTLD_NODELETE);
}

static void load_best_cpu(const std::string & dir) {
    std::string best;
    int best_score = 0;
    if (DIR * d = opendir(dir.c_str())) {
        while (dirent * e = readdir(d)) {
            std::string name = e->d_name;
            if (name.rfind("libggml-cpu-", 0) != 0 || name.size() < 3 || name.substr(name.size() - 3) != ".so") {
                continue;
            }
            std::string path = dir + "/" + name;
            void * h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!h) {
                continue;
            }
            auto score_fn = (int (*)()) dlsym(h, "ggml_backend_score");
            int score = score_fn ? score_fn() : 0;
            dlclose(h);
            if (score > best_score) {
                best_score = score;
                best = path;
            }
        }
        closedir(d);
    }
    if (best.empty() && file_exists(dir + "/libggml-cpu.so")) {
        best = dir + "/libggml-cpu.so";
    }
    if (!best.empty()) {
        pin(best);
        ggml_backend_load(best.c_str());
    }
}
#endif

std::vector<std::string> mv_load_backends(const std::string & dir, const std::string & gpu) {
#ifdef GGML_BACKEND_DL
    if (!g_cpu_loaded) {
        load_best_cpu(dir);
        g_cpu_loaded = true;
    }
    std::stringstream ss(gpu);
    std::string name;
    while (std::getline(ss, name, ',')) {
        if (name.empty() || g_loaded.count(name)) {
            continue;
        }
        std::string path = dir + "/libggml-" + name + ".so";
        if (file_exists(path)) {
            pin(path);
            ggml_backend_load(path.c_str());   // returns NULL (no crash) when the driver is missing
        }
        g_loaded.insert(name);
    }
#else
    (void) dir;
    (void) gpu;
#endif
    std::vector<std::string> out;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * type = "other";
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU: type = "cpu"; break;
            case GGML_BACKEND_DEVICE_TYPE_GPU: type = "gpu"; break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU: type = "igpu"; break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: type = "accel"; break;
            default: break;
        }
        out.push_back(std::string(ggml_backend_dev_name(dev)) + "\t" + ggml_backend_dev_description(dev) + "\t" + type +
                      "\t" + ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)));
    }
    return out;
}

std::string mv_cpu_features() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
    if (!reg) {
        return "";
    }
    struct feature { const char * name; const char * value; };
    auto get = (const feature * (*)(ggml_backend_reg_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    if (!get) {
        return "";
    }
    std::string out;
    for (const feature * f = get(reg); f && f->name; f++) {
        if (!out.empty()) {
            out += " ";
        }
        out += f->name;
        if (f->value && std::string(f->value) != "1") {
            out += std::string("=") + f->value;
        }
    }
    return out;
}

int mv_gpu_index(const std::string & name) {
    int idx = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        auto t = ggml_backend_dev_type(dev);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            if (name == ggml_backend_dev_name(dev)) {
                return idx;
            }
            idx++;
        }
    }
    return -1;
}
