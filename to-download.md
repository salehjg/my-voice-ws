# to-download.md: everything that still has to be downloaded

Nothing has been downloaded yet, neither Python packages nor model weights. There are two steps,
both needing internet. After that, the translator runs fully offline.

| Step | What | Size | Command |
|---|---|---|---|
| 1 | Python env `myvoice` (torch + CUDA libs, faster-whisper, llama.cpp, Piper, …) | ~5 GB | `bash scripts/setup_env.sh` |
| 2 | Model weights into `./models` (Whisper, TranslateGemma, two Piper voices) | ~4.9 GB | `bash scripts/download_models.sh` |
| 3 | Verify (offline) | none | `./run.sh check && ./run.sh selftest` |

> **Disk:** the root partition had only about 43 GB free (96 % used) when this was written. Budget
> ~10 GB for the defaults. Both scripts use `pip --no-cache-dir` / direct-to-folder downloads so
> nothing is duplicated in caches.

> **Toolchain:** every script sources `/home/saleh/00_software/dir_install/setvars.all.sh` first (via
> `env.sh`), as required before using PyTorch/CUDA. If you run commands by hand, always start with
> `source env.sh`, or at least `source /home/saleh/00_software/dir_install/setvars.all.sh`.

---

## Step 1: Python environment (~5 GB)

```bash
cd /workspace/02_ws/my-voice-ws
bash scripts/setup_env.sh                       # default (llama.cpp backend, compiled for this machine)
# options:
#   --with-transformers   also install transformers/accelerate/bitsandbytes (HF-checkpoint backend)
#   --llama-prebuilt      use the prebuilt CUDA wheel instead of compiling (only works on AVX-512 CPUs)
```

What it does, in case you prefer to type it yourself:

```bash
source /home/saleh/00_software/dir_install/setvars.all.sh
conda create -y -n myvoice -c conda-forge python=3.12 pip
conda activate myvoice
pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cu129
pip install --no-cache-dir -r requirements.txt
# llama-cpp-python: compiled for this CPU + sm_89 (the exact CMAKE_ARGS are in scripts/setup_env.sh)
NV=/home/saleh/00_software/dir_install/nvhpc25.7/Linux_x86_64/25.7
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDACXX=$NV/cuda/12.9/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on ..." \
    pip install --no-cache-dir --no-binary llama-cpp-python llama-cpp-python==0.3.36
# optional transformers backend:
pip install --no-cache-dir -r requirements-transformers.txt
```

| Package | Why |
|---|---|
| `torch==2.9.1` (cu129) | Silero VAD, optional transformers backend. Its `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` (cuDNN 9) wheels are also what CTranslate2 and llama.cpp load (`env.sh` puts them on `LD_LIBRARY_PATH`). This torch build is the one that already works on this GPU in `py312ml`. |
| `faster-whisper==1.2.1`, `ctranslate2` | Whisper ASR on CUDA (needs CUDA 12 + cuDNN 9) |
| `llama-cpp-python==0.3.36` (compiled locally, CUDA 12.9) | runs the GGUF translation LLM on the GPU |
| `piper-tts==1.8.0` (+ `onnxruntime`) | neural TTS, one voice per language (uses system `espeak-ng`, already installed) |
| `silero-vad==6.2.3` | voice activity detection; **the model is bundled in the wheel**, nothing else to download |
| `sounddevice` | microphone/speaker via PortAudio (system `portaudio`, already installed) |
| `jinja2`, `scipy`, `numpy` | chat-template rendering, resampling |
| `huggingface_hub` | the `hf` CLI for manual/gated downloads (the step-2 script itself uses `curl`) |
| `pytest` | unit tests |

The setup script finishes by running `./run.sh check`. At that point every package line should be
`ok` and only the `model:` / `voice:` lines should fail.

**Why llama-cpp-python is compiled locally.** The prebuilt CUDA wheel
(`--llama-prebuilt`, from `abetlen.github.io/llama-cpp-python/whl/cu125`) contains AVX-512 CPU
code. This laptop's i9-13950HX has no AVX-512, so the wheel crashes with
`illegal hardware instruction (core dumped)` as soon as llama.cpp starts. The default build uses
`-DGGML_NATIVE=on` for this CPU (AVX2/AVX-VNNI), targets sm_89 with `nvcc` 12.9 from NVHPC, and
uses `/usr/bin/g++-13` as host compiler, because nvcc 12.9 rejects the default GCC 15/16. CUDA 12.9's
`crt/math_functions.h` also clashes with Arch's glibc 2.44 (`exception specification is
incompatible with that of previous function "sinpi"`; CUDA 13 fixes this). So
`scripts/patch_cuda_glibc.sh` creates `build/cuda-12.9-glibc/`, a symlink copy of the NVHPC toolkit
with that one header patched (`noexcept` added to six declarations), and the build uses that. The
NVHPC install is not modified. The build takes roughly 5–15 minutes. To rebuild after a failure, run `bash scripts/setup_env.sh` again; installed
packages are skipped.

---

## Step 2: Models (~4.9 GB with the defaults)

```bash
bash scripts/download_models.sh                     # defaults below
bash scripts/download_models.sh --llm translategemma-q4   # smaller LLM instead (see alternatives)
bash scripts/download_models.sh --extra-voices      # + 4 more voices
bash scripts/download_models.sh --only voices       # just one group: whisper | llm | voices
```

The script downloads over plain HTTPS with `curl` and can be resumed. If the connection drops, it
retries from where it stopped. If it gives up or you press Ctrl+C, run the same command again: the
partial file (`*.part`) is continued and finished files are skipped. Each file is checked against
the size and sha256 published on the Hub before it is moved into place.

### Default set

| Role | Model | Hugging Face repo | File(s) | Size | License |
|---|---|---|---|---|---|
| Speech recognition | Whisper large-v3-turbo, CTranslate2 fp16 | [`mobiuslabsgmbh/faster-whisper-large-v3-turbo`](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo) | whole repo (`model.bin`, `config.json`, `tokenizer.json`, `vocabulary.json`, `preprocessor_config.json`) | 1.62 GB | MIT |
| Translation (LLM) | Google TranslateGemma 4B, GGUF Q6_K | [`mradermacher/translategemma-4b-it-GGUF`](https://huggingface.co/mradermacher/translategemma-4b-it-GGUF) | `translategemma-4b-it.Q6_K.gguf` | 3.19 GB | Gemma Terms of Use |
| Italian voice | Piper `it_IT-paola-medium` | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices) | `it/it_IT/paola/medium/it_IT-paola-medium.onnx` + `.onnx.json` | 64 MB | see the voice's `MODEL_CARD` |
| English voice | Piper `en_US-lessac-high` | `rhasspy/piper-voices` | `en/en_US/lessac/high/en_US-lessac-high.onnx` + `.onnx.json` | 114 MB | see the voice's `MODEL_CARD` |
| VAD | Silero VAD | (inside the `silero-vad` pip wheel) | none | none | MIT |

Equivalent manual commands (from the project root). `HF_HUB_DISABLE_XET=1` makes `hf` use plain
HTTP. Its default Xet transfer can abort on a slow or unstable line with
`CAS Client Error: ... error decoding response body`.

```bash
source env.sh && export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1   # env.sh enables offline mode; turn it off just for downloading

hf download mobiuslabsgmbh/faster-whisper-large-v3-turbo \
    --local-dir models/whisper/faster-whisper-large-v3-turbo

hf download mradermacher/translategemma-4b-it-GGUF translategemma-4b-it.Q6_K.gguf \
    --local-dir models/llm

hf download rhasspy/piper-voices \
    it/it_IT/paola/medium/it_IT-paola-medium.onnx  it/it_IT/paola/medium/it_IT-paola-medium.onnx.json \
    en/en_US/lessac/high/en_US-lessac-high.onnx    en/en_US/lessac/high/en_US-lessac-high.onnx.json \
    --local-dir models/piper

# or any single file with curl (-C - resumes a partial download):
curl -L -C - -o models/llm/translategemma-4b-it.Q6_K.gguf \
    https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q6_K.gguf
```

Resulting layout. These are exactly the paths `config.toml` points to:

```
models/
├── whisper/faster-whisper-large-v3-turbo/{model.bin,config.json,tokenizer.json,vocabulary.json,preprocessor_config.json}
├── llm/translategemma-4b-it.Q6_K.gguf
└── piper/
    ├── it/it_IT/paola/medium/it_IT-paola-medium.onnx(.json)
    └── en/en_US/lessac/high/en_US-lessac-high.onnx(.json)
```

Exact sizes in bytes, to check that a download is complete (`ls -l`):
`model.bin` 1617884929 · `translategemma-4b-it.Q6_K.gguf` 3190755840 ·
`it_IT-paola-medium.onnx` 63511038 · `en_US-lessac-high.onnx` 113895201.

Why these:

- **Whisper large-v3-turbo**: near large-v3 accuracy for Italian and English at a fraction of the
  decode cost (4 decoder layers). Its language ID is what makes the "automatic" direction work.
- **TranslateGemma 4B**: Google's translation-tuned Gemma 3 (Jan 2026). The technical report has
  explicit English→Italian results, where it clearly beats plain Gemma 3 4B. Q6_K is close to full
  quality and leaves VRAM for Whisper. The GGUF is a community conversion, because the original
  `google/translategemma-4b-it` repo is gated. The Gemma Terms of Use still apply.
- **Piper**: real-time on CPU, so the GPU is left for ASR + LLM. It is small and fully offline.

### Alternatives (optional)

**Translation LLM.** Pick one with `--llm <name>`. The script prints the config lines to change.

| `--llm` | Repo / file | Size | `prompt_style` | Notes |
|---|---|---|---|---|
| `translategemma-q6` (default) | `mradermacher/translategemma-4b-it-GGUF` / `translategemma-4b-it.Q6_K.gguf` | 3.19 GB | `translategemma` | best quality/VRAM balance |
| `translategemma-q4` | same repo / `translategemma-4b-it.Q4_K_M.gguf` | 2.49 GB | `translategemma` | lowest VRAM, slightly lower quality |
| `translategemma-q8` | same repo / `translategemma-4b-it.Q8_0.gguf` | 4.13 GB | `translategemma` | max 4B quality; consider `asr.compute_type=int8_float16` |
| `qwen3.5-4b` | `unsloth/Qwen3.5-4B-GGUF` / `Qwen3.5-4B-Q5_K_M.gguf` | 3.14 GB | `chat` | Apache-2.0 general model; uses the last turns as context |
| `gemma4-e4b` | `ggml-org/gemma-4-E4B-it-GGUF` / `gemma-4-E4B-it-Q4_0.gguf` | 4.59 GB | `chat` | Apache-2.0; tight on 8 GB, use `asr.compute_type=int8_float16` |

**No-download fallback.** `Qwen/Qwen2.5-7B-Instruct` is already complete in `~/.cache/huggingface`
(15 GB of safetensors). With the transformers backend (`setup_env.sh --with-transformers`) it
runs in 4-bit NF4 (~5.5 GB VRAM):

```bash
./run.sh live --set translation.backend=transformers --set translation.model=Qwen/Qwen2.5-7B-Instruct \
              --set translation.prompt_style=chat --set asr.compute_type=int8_float16
```

**Original TranslateGemma checkpoint** (transformers backend, gated, ~8.6 GB). Accept the license
at <https://huggingface.co/google/translategemma-4b-it>, then:

```bash
source env.sh && export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1
hf auth login
hf download google/translategemma-4b-it --local-dir models/llm/translategemma-4b-it
# run with: --set translation.backend=transformers --set translation.model=models/llm/translategemma-4b-it
```

**ASR.** For maximum accuracy use `Systran/faster-whisper-large-v3` (~3.1 GB fp16, slower, about
1.5 GB more VRAM, so pair it with `translategemma-q4` or `compute_type=int8_float16`):
`hf download Systran/faster-whisper-large-v3 --local-dir models/whisper/faster-whisper-large-v3`,
then set `asr.model`.

**Voices** (`--extra-voices`). Switch with `[tts.voices]` in `config.toml`. All are in
`rhasspy/piper-voices`:

| Voice | Path in repo | Size |
|---|---|---|
| Italian, female, high | `it/it_IT/serena/high/it_IT-serena-high.onnx` | 114 MB |
| Italian, male, x_low | `it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx` | 28 MB |
| US English, male, high | `en/en_US/ryan/high/en_US-ryan-high.onnx` | 121 MB |
| British English, female, high | `en/en_GB/cori/high/en_GB-cori-high.onnx` | 114 MB |

(Each needs its `.onnx.json` too; the script fetches both.)

---

## Step 3: Verify (offline)

```bash
./run.sh check        # packages, CUDA libs, model files: all lines should be ok
./run.sh selftest     # Piper speaks an IT and an EN sentence -> Whisper -> LLM -> Piper; prints timings + VRAM
./run.sh live         # the real thing (headphones recommended: add --headphones)
```

## Runtime is offline

`env.sh` (used by `run.sh`) sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Every model is
then loaded from a local path (`local_files_only=True` for Whisper and transformers; plain file
paths for GGUF and Piper), and the Silero VAD comes from inside its wheel. Nothing is fetched
while translating, so you can unplug the network after steps 1–2.
