// Plain C++ API of the native part (no JNI), so it can also be built and tested on the laptop.
#pragma once

#include <string>
#include <vector>

// ---- ggml backends
// Load the best CPU variant from `dir` and the GPU backends named in `gpu` ("vulkan,opencl").
// Returns one line per device: "name\tdescription\ttype\tbackend".
std::vector<std::string> mv_load_backends(const std::string & dir, const std::string & gpu);
std::string mv_cpu_features();
// Index of device `name` among GPU/iGPU devices (whisper.cpp's gpu_device), or -1.
int mv_gpu_index(const std::string & name);

// ---- whisper.cpp
void * mv_whisper_init(const std::string & model, const std::string & device, bool flash_attn);
// language "" = detect; probs (whisper_lang_max_id()+1 floats) receives detection probabilities.
// Returns "lang\ttext"; ok=false on failure.
std::string mv_whisper_transcribe(void * ctx, const float * pcm, int n, const std::string & language, int threads,
                                  const std::string & prompt, int audio_ctx, float * probs, int n_probs, bool & ok);
int mv_whisper_lang_id(const std::string & code);
int mv_whisper_lang_count();
void mv_whisper_free(void * ctx);

// ---- llama.cpp
void * mv_llama_init(const std::string & model, const std::string & device, int gpu_layers, int n_ctx, int threads);
std::string mv_llama_complete(void * ctx, const std::string & prompt, int max_tokens,
                              const std::vector<std::string> & stops, bool & ok);
void mv_llama_free(void * ctx);

// ---- espeak-ng (Piper phonemes)
int mv_espeak_init(const std::string & data_dir);
int mv_espeak_set_voice(const std::string & voice);
// One entry per clause: phonemes \x01 terminator \x01 end-of-sentence ("0"/"1").
std::vector<std::string> mv_espeak_phonemize(const std::string & text);

// ---- device facts (Android)
std::vector<std::string> mv_vulkan_devices();
std::vector<std::string> mv_opencl_devices();
std::vector<std::string> mv_nnapi_devices();
