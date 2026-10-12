// llama.cpp: greedy completion for the translation prompts (TranslateGemma / Gemma chat).

#include "mv_native.h"

#include "ggml-backend.h"
#include "llama.h"

#include <mutex>

struct mv_llama {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_sampler * smpl = nullptr;
    int n_ctx = 0;
};

static std::once_flag g_backend_once;

void * mv_llama_init(const std::string & model_path, const std::string & device, int gpu_layers, int n_ctx, int threads) {
    std::call_once(g_backend_once, [] { llama_backend_init(); });
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devices[2] = {nullptr, nullptr};
    if (!device.empty()) {
        devices[0] = ggml_backend_dev_by_name(device.c_str());
        if (!devices[0]) {
            return nullptr;
        }
        mp.n_gpu_layers = gpu_layers < 0 ? 999 : gpu_layers;
    } else {
        mp.n_gpu_layers = 0;   // CPU only: an empty device list keeps everything off the GPU
    }
    mp.devices = devices;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;   // file-backed pages: Android can reclaim them under pressure
    auto * h = new mv_llama();
    h->model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!h->model) {
        delete h;
        return nullptr;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.no_perf = true;
    h->ctx = llama_init_from_model(h->model, cp);
    if (!h->ctx) {
        llama_model_free(h->model);
        delete h;
        return nullptr;
    }
    h->vocab = llama_model_get_vocab(h->model);
    h->n_ctx = n_ctx;
    h->smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(h->smpl, llama_sampler_init_greedy());
    return h;
}

std::string mv_llama_complete(void * handle, const std::string & prompt, int max_tokens,
                              const std::vector<std::string> & stops, bool & ok) {
    auto * h = (mv_llama *) handle;
    ok = false;
    llama_memory_clear(llama_get_memory(h->ctx), true);
    llama_sampler_reset(h->smpl);
    int n = -llama_tokenize(h->vocab, prompt.c_str(), (int) prompt.size(), nullptr, 0, true, true);
    if (n <= 0) {
        return "";
    }
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(h->vocab, prompt.c_str(), (int) prompt.size(), tokens.data(), n, true, true) < 0) {
        return "";
    }
    if (n >= h->n_ctx - 8) {
        return "";   // prompt too long for the context
    }
    max_tokens = std::min(max_tokens, h->n_ctx - n - 1);
    if (llama_decode(h->ctx, llama_batch_get_one(tokens.data(), n)) != 0) {
        return "";
    }
    std::string out;
    char buf[256];
    for (int i = 0; i < max_tokens; i++) {
        llama_token tok = llama_sampler_sample(h->smpl, h->ctx, -1);
        if (llama_vocab_is_eog(h->vocab, tok)) {
            break;
        }
        int len = llama_token_to_piece(h->vocab, tok, buf, sizeof(buf), 0, true);
        if (len > 0) {
            out.append(buf, len);
        }
        bool stop = false;
        for (const auto & s : stops) {
            size_t at = out.find(s);
            if (at != std::string::npos) {
                out.resize(at);
                stop = true;
                break;
            }
        }
        if (stop) {
            break;
        }
        if (llama_decode(h->ctx, llama_batch_get_one(&tok, 1)) != 0) {
            return "";
        }
    }
    ok = true;
    return out;
}

void mv_llama_free(void * handle) {
    auto * h = (mv_llama *) handle;
    if (!h) {
        return;
    }
    if (h->smpl) {
        llama_sampler_free(h->smpl);
    }
    if (h->ctx) {
        llama_free(h->ctx);
    }
    if (h->model) {
        llama_model_free(h->model);
    }
    delete h;
}
