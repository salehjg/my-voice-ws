# to-download.md: what has to be downloaded, and how

Two steps need internet: the Python environment, then the model files. After that, the translator
runs fully offline. **On this laptop both are done** (Oct 2026); this file is for a fresh setup or a
rebuild. Both commands can be re-run at any time: packages and files that are already there are
skipped.

| Step | What | Size | Command |
|---|---|---|---|
| 1 | Python env `myvoice` (torch + CUDA libs, faster-whisper, llama.cpp, Piper, Kokoro, …) | ~5 GB | `./myvoice.py setup` |
| 2 | Model weights into `./models` (Whisper, TranslateGemma, 7 Piper voices, Kokoro) | ~5.8 GB | `./myvoice.py download` |
| 3 | Verify (offline) | none | `./myvoice.py check && ./myvoice.py selftest` |

> **Disk:** the root partition had only about 43 GB free (96 % used) when this was written. Budget
> ~10 GB for the defaults. Both commands use `pip --no-cache-dir` / direct-to-folder downloads so
> nothing is duplicated in caches.

> **Toolchain:** `./myvoice.py` sources `/home/saleh/00_software/dir_install/setvars.all.sh` by itself
> before anything else, as required before using PyTorch/CUDA (any `python3` can start it). If you run
> commands by hand, always start with `source /home/saleh/00_software/dir_install/setvars.all.sh && conda activate myvoice`.

---

## Step 1: Python environment (~5 GB)

```bash
cd /workspace/02_ws/my-voice-ws
./myvoice.py setup                              # default (llama.cpp backend, compiled for this machine)
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
# llama-cpp-python: compiled for this CPU + sm_89 (the exact CMAKE_ARGS are in src/myvoice/install.py)
NV=/home/saleh/00_software/dir_install/nvhpc25.7/Linux_x86_64/25.7
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDACXX=$NV/cuda/12.9/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on ..." \
    pip install --no-cache-dir --no-binary llama-cpp-python llama-cpp-python==0.3.36
# optional transformers backend:
pip install --no-cache-dir -r requirements-transformers.txt
```

| Package | Why |
|---|---|
| `torch==2.9.1` (cu129) | Silero VAD, optional transformers backend. Its `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` (cuDNN 9) wheels are also what CTranslate2 and llama.cpp load (`./myvoice.py` puts them on `LD_LIBRARY_PATH`). This torch build is the one that already works on this GPU in `py312ml`. |
| `faster-whisper==1.2.1`, `ctranslate2` | Whisper ASR on CUDA (needs CUDA 12 + cuDNN 9) |
| `llama-cpp-python==0.3.36` (compiled locally, CUDA 12.9) | runs the GGUF translation LLM on the GPU |
| `piper-tts==1.8.0` (+ `onnxruntime`) | neural TTS, one voice per language (uses system `espeak-ng`, already installed) |
| `silero-vad==6.2.3` | voice activity detection; **the model is bundled in the wheel**, nothing else to download |
| `sounddevice` | microphone/speaker via PortAudio (system `portaudio`, already installed) |
| `jinja2`, `scipy`, `numpy` | chat-template rendering, resampling |
| `huggingface_hub` | the `hf` CLI for manual/gated downloads (`./myvoice.py download` itself uses plain HTTPS) |
| `rich`, `typer` | the terminal UI and the command line |
| `kokoro-onnx` | Kokoro voices |
| `pytest` | unit tests (`./myvoice.py test`) |

`./myvoice.py setup` finishes by running `check`. At that point every package line should be
`ok` and only the `model:` / `voice:` lines should fail.

**Why llama-cpp-python is compiled locally.** The prebuilt CUDA wheel
(`--llama-prebuilt`, from `abetlen.github.io/llama-cpp-python/whl/cu125`) contains AVX-512 CPU
code. This laptop's i9-13950HX has no AVX-512, so the wheel crashes with
`illegal hardware instruction (core dumped)` as soon as llama.cpp starts. The default build uses
`-DGGML_NATIVE=on` for this CPU (AVX2/AVX-VNNI), targets sm_89 with `nvcc` 12.9 from NVHPC, and
uses `/usr/bin/g++-13` as host compiler, because nvcc 12.9 rejects the default GCC 15/16. CUDA 12.9's
`crt/math_functions.h` also clashes with Arch's glibc 2.44 (`exception specification is
incompatible with that of previous function "sinpi"`; CUDA 13 fixes this). So
setup (`patch_cuda_glibc` in `src/myvoice/install.py`) creates `build/cuda-12.9-glibc/`, a symlink copy of the NVHPC toolkit
with that one header patched (`noexcept` added to six declarations), and the build uses that. The
NVHPC install is not modified. The build takes roughly 5–15 minutes. To rebuild after a failure, run `./myvoice.py setup` again;
installed packages are skipped.

---

## Step 2: Models (~5.8 GB)

```bash
./myvoice.py download                          # everything below
./myvoice.py download --only voices            # just one group: whisper | llm | voices | kokoro
./myvoice.py download --llm translategemma-q4  # a different LLM (see alternatives)
./myvoice.py download --asr large-v3           # + optional Whisper large-v3 (3.1 GB, see below)
```

Downloads go over plain HTTPS and can be resumed. If the connection drops, it retries from where
it stopped. If it gives up or you press Ctrl+C, run the same command again: the
partial file (`*.part`) is continued and **finished files are skipped** (a re-run with everything
present takes a second and downloads nothing). Each file is checked against its size and sha256
before it is moved into place.

### Default set

| Role | Model | Source | File(s) | Size | License |
|---|---|---|---|---|---|
| Speech recognition | Whisper large-v3-turbo, CTranslate2 fp16 | [`mobiuslabsgmbh/faster-whisper-large-v3-turbo`](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo) | `model.bin`, `config.json`, `tokenizer.json`, `vocabulary.json`, `preprocessor_config.json` | 1.62 GB | MIT |
| Translation (LLM) | Google TranslateGemma 4B, GGUF Q6_K | [`mradermacher/translategemma-4b-it-GGUF`](https://huggingface.co/mradermacher/translategemma-4b-it-GGUF) | `translategemma-4b-it.Q6_K.gguf` | 3.19 GB | Gemma Terms of Use |
| Voices (default) | Kokoro-82M, ONNX | [`thewh1teagle/kokoro-onnx` release `model-files-v1.1`](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1) | `kokoro-v1.0.onnx`, `voices-v1.0.bin` | 354 MB | Apache-2.0 |
| Voices (light) | 7 Piper voices (see below) | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices) | `.onnx` + `.onnx.json` each | 618 MB | see each voice's `MODEL_CARD` |
| VAD | Silero VAD | (inside the `silero-vad` pip wheel) | none | none | MIT |

Piper voices in the default set (path in `rhasspy/piper-voices`):

| Voice | Path | Size |
|---|---|---|
| Italian, female (Paola) | `it/it_IT/paola/medium/it_IT-paola-medium.onnx` | 64 MB |
| Italian, female (Serena) | `it/it_IT/serena/high/it_IT-serena-high.onnx` | 114 MB |
| Italian, male (Riccardo, low quality) | `it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx` | 28 MB |
| US English, female (Lessac) | `en/en_US/lessac/high/en_US-lessac-high.onnx` | 114 MB |
| US English, male (Ryan) | `en/en_US/ryan/high/en_US-ryan-high.onnx` | 121 MB |
| British English, female (Cori) | `en/en_GB/cori/high/en_GB-cori-high.onnx` | 114 MB |
| British English, male (Alan) | `en/en_GB/alan/medium/en_GB-alan-medium.onnx` | 63 MB |

Equivalent manual commands (from the project root). `HF_HUB_DISABLE_XET=1` makes `hf` use plain
HTTP. Its default Xet transfer can abort on a slow or unstable line with
`CAS Client Error: ... error decoding response body`.

```bash
source /home/saleh/00_software/dir_install/setvars.all.sh && conda activate myvoice && export HF_HUB_DISABLE_XET=1

hf download mobiuslabsgmbh/faster-whisper-large-v3-turbo \
    --local-dir models/whisper/faster-whisper-large-v3-turbo
hf download mradermacher/translategemma-4b-it-GGUF translategemma-4b-it.Q6_K.gguf --local-dir models/llm
hf download rhasspy/piper-voices \
    it/it_IT/paola/medium/it_IT-paola-medium.onnx it/it_IT/paola/medium/it_IT-paola-medium.onnx.json \
    --local-dir models/piper          # ...and the same for the other voices in the table

# Kokoro (GitHub release); -C - resumes a partial download
curl -L -C - -o models/kokoro/kokoro-v1.0.onnx \
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx
curl -L -C - -o models/kokoro/voices-v1.0.bin \
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin
```

Resulting layout (the paths `config.toml` uses):

```
models/
├── whisper/faster-whisper-large-v3-turbo/{model.bin,config.json,tokenizer.json,vocabulary.json,preprocessor_config.json}
├── llm/translategemma-4b-it.Q6_K.gguf
├── kokoro/{kokoro-v1.0.onnx,voices-v1.0.bin}
└── piper/{it,en}/<locale>/<name>/<quality>/<voice>.onnx(.json)
```

Exact sizes in bytes / sha256, to check a download by hand (`ls -l`, `sha256sum`):
`model.bin` 1617884929 · `translategemma-4b-it.Q6_K.gguf` 3190755840 ·
`kokoro-v1.0.onnx` 325505369 (`beb0d184…f0df3a`) · `voices-v1.0.bin` 28214398 (`bca610b8…f1fbf7d`).

Why these:

- **Whisper large-v3-turbo**: near large-v3 accuracy for Italian and English at a fraction of the
  decode cost (4 decoder layers).
- **TranslateGemma 4B**: Google's translation-tuned Gemma 3 (Jan 2026). One model translates both
  ways (Italian→English and English→Italian). Its technical report has explicit English→Italian
  results, where it clearly beats plain Gemma 3 4B. Q6_K is close to full quality and leaves VRAM
  for Whisper. The GGUF is a community conversion, because the original
  `google/translategemma-4b-it` repo is gated. The Gemma Terms of Use still apply.
- **Kokoro + Piper**: both run on the CPU, leaving the GPU for Whisper + the LLM. In a round-trip
  test (voice speaks → Whisper transcribes → compare), Kokoro was the most intelligible in Italian.
  Piper voices are lighter and faster. Kokoro also has many more English voices
  (`./myvoice.py voices --all`).

### Alternatives (optional)

**Translation LLM.** Pick one with `./myvoice.py download --llm <name>`. It prints the config lines to change.

| `--llm` | Repo / file | Size | `prompt_style` | Notes |
|---|---|---|---|---|
| `translategemma-q6` (default) | `mradermacher/translategemma-4b-it-GGUF` / `translategemma-4b-it.Q6_K.gguf` | 3.19 GB | `translategemma` | best quality/VRAM balance |
| `translategemma-q4` | same repo / `translategemma-4b-it.Q4_K_M.gguf` | 2.49 GB | `translategemma` | lowest VRAM, slightly lower quality |
| `translategemma-q8` | same repo / `translategemma-4b-it.Q8_0.gguf` | 4.13 GB | `translategemma` | max 4B quality; consider `asr.compute_type=int8_float16` |
| `qwen3.5-4b` | `unsloth/Qwen3.5-4B-GGUF` / `Qwen3.5-4B-Q5_K_M.gguf` | 3.14 GB | `chat` | Apache-2.0 general model; uses the last turns as context |
| `gemma4-e4b` | `ggml-org/gemma-4-E4B-it-GGUF` / `gemma-4-E4B-it-Q4_0.gguf` | 4.59 GB | `chat` | Apache-2.0; tight on 8 GB, use `asr.compute_type=int8_float16` |

**No-download fallback.** `Qwen/Qwen2.5-7B-Instruct` is already complete in `~/.cache/huggingface`
(15 GB of safetensors). With the transformers backend (`./myvoice.py setup --with-transformers`) it
runs in 4-bit NF4 (~5.5 GB VRAM):

```bash
./myvoice.py --set translation.backend=transformers --set translation.model=Qwen/Qwen2.5-7B-Instruct \
             --set translation.prompt_style=chat --set asr.compute_type=int8_float16
```

**Original TranslateGemma checkpoint** (transformers backend, gated, ~8.6 GB). Accept the license
at <https://huggingface.co/google/translategemma-4b-it>, then:

```bash
source /home/saleh/00_software/dir_install/setvars.all.sh && conda activate myvoice && export HF_HUB_DISABLE_XET=1
hf auth login
hf download google/translategemma-4b-it --local-dir models/llm/translategemma-4b-it
# run with: --set translation.backend=transformers --set translation.model=models/llm/translategemma-4b-it
```

**ASR: Whisper large-v3** (`--asr large-v3`, `Systran/faster-whisper-large-v3`, 3.1 GB). Downloaded
on this laptop, but **not used by default**. On 192 accented-English test sentences (24 non-native
speakers, background noise) it scored 4.3–4.5 % word errors against turbo's 4.6 %, while being
50–60 % slower. To try it anyway:
`./myvoice.py --set asr.model=models/whisper/faster-whisper-large-v3 --set asr.compute_type=int8_float16`.
To free the space: `rm -rf models/whisper/faster-whisper-large-v3`.

---

## Step 3: Verify (offline)

```bash
./myvoice.py check      # packages, CUDA libs, model files: all lines should be ok
./myvoice.py selftest   # a voice speaks an IT and an EN sentence -> Whisper -> LLM -> voice; prints timings + VRAM
./myvoice.py            # the conversation app (see README)
```

## Runtime is offline

`./myvoice.py` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for the app. Every model is
then loaded from a local path (`local_files_only=True` for Whisper and transformers; plain file
paths for GGUF, Piper and Kokoro), the Silero VAD comes from inside its wheel, and Kokoro's
phonemizer (espeak-ng) is bundled with its pip package. Nothing is fetched
while translating, so you can unplug the network after steps 1–2.
