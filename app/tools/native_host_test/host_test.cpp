// Command-line driver for the app's native functions (laptop build, CPU only).
//   host_test espeak  DATA_DIR VOICE TEXT
//   host_test whisper MODEL WAV16K LANG|auto THREADS AUDIO_CTX
//   host_test llama   MODEL PROMPT_FILE MAX_TOKENS THREADS
#include "mv_native.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<float> read_wav16(const char * path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t pos = 12;
    int channels = 1, bits = 16;
    while (pos + 8 <= b.size()) {
        std::string id(&b[pos], 4);
        uint32_t size;
        memcpy(&size, &b[pos + 4], 4);
        if (id == "fmt ") {
            channels = *(int16_t *) &b[pos + 10];
            bits = *(int16_t *) &b[pos + 22];
        } else if (id == "data") {
            std::vector<float> out;
            const int16_t * s = (const int16_t *) &b[pos + 8];
            size_t frames = size / 2 / channels;
            for (size_t i = 0; i < frames && bits == 16; i++) {
                float sum = 0;
                for (int c = 0; c < channels; c++) sum += s[i * channels + c] / 32768.0f;
                out.push_back(sum / channels);
            }
            return out;
        }
        pos += 8 + size + (size & 1);
    }
    return {};
}

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: see host_test.cpp\n");
        return 2;
    }
    std::string cmd = argv[1];
    mv_load_backends("", "");
    if (cmd == "espeak" && argc == 5) {
        if (mv_espeak_init(argv[2]) != 0 || mv_espeak_set_voice(argv[3]) != 0) {
            fprintf(stderr, "espeak init failed\n");
            return 1;
        }
        for (auto & c : mv_espeak_phonemize(argv[4])) {
            for (auto & ch : c) if (ch == '\x01') ch = '|';
            printf("%s\n", c.c_str());
        }
        return 0;
    }
    if (cmd == "whisper" && argc == 7) {
        void * ctx = mv_whisper_init(argv[2], "", true);
        if (!ctx) return 1;
        auto pcm = read_wav16(argv[3]);
        std::string lang = std::string(argv[4]) == "auto" ? "" : argv[4];
        std::vector<float> probs(mv_whisper_lang_count(), 0.0f);
        bool ok = false;
        double t0 = now();
        std::string out = mv_whisper_transcribe(ctx, pcm.data(), (int) pcm.size(), lang, atoi(argv[5]), "", atoi(argv[6]),
                                                probs.data(), (int) probs.size(), ok);
        double dt = now() - t0;
        printf("%s\n", out.c_str());
        printf("p_it=%.3f p_en=%.3f p_es=%.3f seconds=%.2f ok=%d\n", probs[mv_whisper_lang_id("it")], probs[mv_whisper_lang_id("en")],
               probs[mv_whisper_lang_id("es")], dt, ok);
        mv_whisper_free(ctx);
        return ok ? 0 : 1;
    }
    if (cmd == "llama" && argc == 6) {
        std::ifstream f(argv[3]);
        std::stringstream ss;
        ss << f.rdbuf();
        void * ctx = mv_llama_init(argv[2], "", 0, 1024, atoi(argv[5]));
        if (!ctx) return 1;
        bool ok = false;
        double t0 = now();
        std::string out = mv_llama_complete(ctx, ss.str(), atoi(argv[4]), {"<end_of_turn>"}, ok);
        printf("%s\nseconds=%.2f ok=%d\n", out.c_str(), now() - t0, ok);
        mv_llama_free(ctx);
        return ok ? 0 : 1;
    }
    fprintf(stderr, "bad arguments\n");
    return 2;
}
