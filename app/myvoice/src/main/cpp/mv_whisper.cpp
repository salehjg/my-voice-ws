// whisper.cpp: transcription with language detection restricted to the caller's pair.
// Built against a patched copy of whisper.cpp (see CMakeLists.txt) that (1) uses the shortened
// audio window for language detection too and (2) hands back the detection probabilities.

#include "mv_native.h"

#include "whisper.h"

#include <cstring>

extern "C" const float * whisper_myvoice_lang_probs(int * n);

void * mv_whisper_init(const std::string & model, const std::string & device, bool flash_attn) {
    whisper_context_params cp = whisper_context_default_params();
    cp.flash_attn = flash_attn;
    if (!device.empty()) {
        int idx = mv_gpu_index(device);
        if (idx < 0) {
            return nullptr;
        }
        cp.use_gpu = true;
        cp.gpu_device = idx;
    } else {
        cp.use_gpu = false;
    }
    return whisper_init_from_file_with_params(model.c_str(), cp);
}

std::string mv_whisper_transcribe(void * h, const float * pcm, int n, const std::string & language, int threads,
                                  const std::string & prompt, int audio_ctx, float * probs, int n_probs, bool & ok) {
    auto * ctx = (whisper_context *) h;
    whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    p.n_threads = threads;
    p.language = language.empty() ? "auto" : language.c_str();
    p.detect_language = false;
    p.translate = false;
    p.no_context = true;
    p.no_timestamps = true;
    p.single_segment = false;
    p.print_progress = false;
    p.print_realtime = false;
    p.print_special = false;
    p.print_timestamps = false;
    p.suppress_blank = true;
    p.suppress_nst = true;
    p.audio_ctx = audio_ctx;
    if (!prompt.empty()) {
        p.initial_prompt = prompt.c_str();
    }
    // whisper.cpp ignores clips under 100 ms; give very short ones a little silence
    std::vector<float> padded;
    if (n < 16000) {
        padded.assign(pcm, pcm + n);
        padded.resize(16000, 0.0f);
        pcm = padded.data();
        n = (int) padded.size();
    }
    if (whisper_full(ctx, p, pcm, n) != 0) {
        ok = false;
        return "";
    }
    ok = true;
    std::string text;
    for (int i = 0; i < whisper_full_n_segments(ctx); i++) {
        const char * s = whisper_full_get_segment_text(ctx, i);
        if (s) {
            if (!text.empty()) {
                text += " ";
            }
            text += s;
        }
    }
    int lang_id = whisper_full_lang_id(ctx);
    const char * lang = lang_id >= 0 ? whisper_lang_str(lang_id) : (language.empty() ? "" : language.c_str());
    if (probs && language.empty()) {
        int m = 0;
        const float * src = whisper_myvoice_lang_probs(&m);
        for (int i = 0; i < n_probs && i < m; i++) {
            probs[i] = src[i];
        }
    }
    return std::string(lang ? lang : "") + "\t" + text;
}

int mv_whisper_lang_id(const std::string & code) {
    return whisper_lang_id(code.c_str());
}

int mv_whisper_lang_count() {
    return whisper_lang_max_id() + 1;
}

void mv_whisper_free(void * ctx) {
    if (ctx) {
        whisper_free((whisper_context *) ctx);
    }
}
