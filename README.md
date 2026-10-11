# myvoice: offline Italian ⇄ English voice translator

Speak Italian and you hear English; speak English and you hear Italian. The direction is chosen
automatically for each sentence. Everything runs locally on the RTX 2000 Ada (8 GB) with no
network.

**First time:** follow [`to-download.md`](to-download.md) (Python env + models), then
`./run.sh check` and `./run.sh selftest`.

## How it works

```
 mic ──► Silero VAD ──► Whisper large-v3-turbo ──► TranslateGemma 4B (LLM) ──► Piper TTS ──► speaker
 16 kHz   (CPU)          faster-whisper / CUDA       llama.cpp / CUDA           (CPU, ONNX)
          cuts the        transcribes + detects       translates into the        Italian or
          stream into     it vs en (restricted        other language             English voice
          utterances      to the pair)
```

| Stage | Component | Where | Notes |
|---|---|---|---|
| VAD | Silero VAD v6 (`silero-vad`) | CPU | an utterance ends after 700 ms of silence (`vad.min_silence_ms`) |
| ASR | Whisper large-v3-turbo via CTranslate2 | GPU, fp16 | language ID is limited to {it, en}; if Whisper picks e.g. Spanish we re-decode as the likelier of the pair; known Whisper hallucinations are filtered |
| MT | TranslateGemma 4B Q6_K (GGUF) via llama-cpp-python | GPU | official TranslateGemma prompt rendered with the model's own chat template; greedy decoding |
| TTS | Piper (`it_IT-paola-medium`, `en_US-lessac-high`) | CPU | faster than real time, keeps the GPU free |

A *cascaded* pipeline (ASR → LLM → TTS) was chosen over end-to-end speech models because each
part is the best available at its size. The LLM gets clean text and can be swapped (Qwen 3.5,
Gemma 4, …) with one config line, and the whole stack fits in 8 GB.

Threads in `live` mode: PortAudio callback → segmenter (VAD) → worker (ASR→MT→TTS) → player.
Translation of sentence *n* overlaps with you speaking sentence *n+1*. The player waits for a
pause before speaking. In **half-duplex** mode (the default) the mic is muted while the
translation plays, so the speaker output isn't re-translated. With headphones use `--headphones`
for full duplex.

### VRAM budget (estimated; `selftest` prints the real number)

| | GB |
|---|---|
| Whisper large-v3-turbo fp16 + work buffers | ~1.9 |
| TranslateGemma 4B Q6_K + 2k KV cache + buffers | ~3.8 |
| CUDA contexts (CTranslate2 + llama.cpp) | ~0.4 |
| **Total** | **~6.1 of 8** |

If memory is tight: `--set asr.compute_type=int8_float16` (−0.7 GB) and/or the Q4_K_M LLM
(−0.7 GB).

## Usage

`run.sh` sources `env.sh`, which sources `/home/saleh/00_software/dir_install/setvars.all.sh`
(required before any PyTorch/CUDA use), activates the `myvoice` conda env and turns on offline
mode. In an interactive shell, `source env.sh` does the same.

```bash
./run.sh check                         # environment + model files (no model loading)
./run.sh devices                       # list audio devices
./run.sh selftest                      # mic-free end-to-end test, writes outputs/selftest_*.wav

./run.sh live                          # auto it⇄en, mic -> speakers (half-duplex)
./run.sh live --headphones             # full duplex
./run.sh live --mode it-en             # fixed direction (skips language detection)
./run.sh live --input-device "USB" --output-device 5

./run.sh file intervista.m4a -o outputs/intervista_en.wav    # translate a recording (+ .jsonl transcript)
./run.sh text "Dov'è la stazione?" --speak                    # typed text -> translation (+ speech)
./run.sh text                                                 # interactive text REPL
```

Every live session writes a JSONL transcript to `outputs/session-*.jsonl`. Each line holds the
source text, the translation, the detected language with its confidence, and per-stage timings.

### Configuration

All settings live in [`config.toml`](config.toml) and are commented there. Any value can be
overridden per run with `--set section.key=value`. The useful ones:

| Setting | Default | When to change |
|---|---|---|
| `general.mode` | `auto` | `it-en` / `en-it` when only one person speaks; it avoids language-ID mistakes on very short phrases |
| `vad.min_silence_ms` | 700 | lower = faster turn-taking but may split sentences; raise it for slow speakers |
| `vad.threshold` | 0.5 | raise (0.6–0.7) in noisy rooms |
| `asr.compute_type` | `float16` | `int8_float16` saves VRAM |
| `asr.initial_prompt` | `""` | names or jargon Whisper should spell right |
| `translation.model` / `prompt_style` | TranslateGemma / `translategemma` | other LLMs use `chat` (see to-download.md) |
| `translation.context_turns` | 2 | `chat` style only: previous exchanges given to the LLM for pronouns/gender agreement |
| `tts.voices.it` / `.en` | paola / lessac-high | any Piper voice (`--extra-voices` downloads 4 more) |
| `tts.length_scale` | 1.0 | 1.1–1.2 for slower, clearer speech |
| `audio.half_duplex` | `true` | `false` with headphones |

### Expected latency

These are estimates, not measurements; `selftest` and the live output print real per-stage
timings. After you stop talking: about 0.7 s of silence detection, then ASR ≈0.2–0.5 s, LLM
≈0.4–1.5 s depending on sentence length, and Piper ≈0.1–0.3 s. That gives roughly **1.5–3 s**
from the end of your sentence to the start of the translation.

## Project layout

```
config.toml               all settings
env.sh                    source me: setvars.all.sh + conda env + CUDA lib paths + offline flags
run.sh                    ./run.sh <command>  (wraps env.sh + python -m myvoice)
to-download.md            what to download and how (env + models)
requirements*.txt         pip requirements (core / optional transformers backend)
scripts/setup_env.sh      creates the conda env (downloads packages)
scripts/download_models.sh downloads models into ./models
src/myvoice/
  __main__.py             CLI: check | devices | live | file | text | selftest
  config.py               typed config, TOML loading, --set overrides
  langs.py                language table, direction logic, typed-text language guess
  audio.py                mic stream, playback, resampling, WAV I/O
  vad.py                  Silero / energy VAD + utterance segmenter
  asr.py                  faster-whisper wrapper, pair-restricted language ID, hallucination filter
  translate.py            prompts, chat-template rendering, llama.cpp + transformers backends
  tts.py                  Piper voices
  pipeline.py             Engine (one turn), LiveSession (threads), file mode, selftest
  check.py                environment / model verification
tests/                    unit tests with fake models (no GPU or downloads needed)
```

## Tests

```bash
source env.sh && python -m pytest -q tests
```

The tests replace the models with fakes. They cover config parsing, direction logic, VAD
segmentation, prompt construction and template rendering, output cleaning, the hallucination
filter, and the threaded live loop including mic muting during playback.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `check`: *ctranslate2 sees no CUDA device* / `libcudnn.so.9` not found | run through `./run.sh` or `source env.sh`; it puts the pip NVIDIA libs on `LD_LIBRARY_PATH` |
| `check`: *llama-cpp-python built WITHOUT GPU support* or *crashed (illegal instruction)* / `illegal hardware instruction (core dumped)` | `bash scripts/setup_env.sh` (compiles llama-cpp-python for this CPU; the prebuilt wheel needs AVX-512) |
| CUDA out of memory | `--set asr.compute_type=int8_float16`, the Q4_K_M LLM, or close other GPU apps |
| It translates its own voice | use headphones, or keep `audio.half_duplex = true` and raise `audio.playback_tail_ms` |
| Sentences cut in half | raise `vad.min_silence_ms` (e.g. 900) |
| Short words get the wrong direction ("sì", "ok") | use a fixed `--mode it-en` / `en-it` |
| Phantom phrases from noise | raise `vad.threshold`; the built-in filter already drops "Sottotitoli a cura di…" etc. |
| `could not open the microphone` | `./run.sh devices`, then `--input-device <index or name>` |
