#!/usr/bin/env bash
# Download the model files into ./models.
#
#   bash scripts/download_models.sh                      # everything needed (~5.8 GB on a fresh machine)
#   bash scripts/download_models.sh --llm translategemma-q4   # smaller LLM (2.5 GB)
#   bash scripts/download_models.sh --llm qwen3.5-4b          # alternative general LLM (needs prompt_style="chat")
#   bash scripts/download_models.sh --asr large-v3            # + Whisper large-v3 (3.1 GB, optional, see README)
#   bash scripts/download_models.sh --only whisper|llm|voices|kokoro
#
# Files that are already there are skipped, so running it again is cheap. Every file is pinned to
# an explicit repo + filename (see to-download.md). Downloads use plain HTTPS via curl and are
# resumable: if the connection drops, the script retries, and if it is killed, running it again
# continues where it stopped. Finished files are checked against their size and sha256.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LLM="translategemma-q6"
ASR=""
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --llm) LLM="$2"; shift 2 ;;
    --asr) ASR="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --extra-voices) shift ;;  # old flag: the extra voices are part of the default set now
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Same environment as everything else (setvars.all.sh + conda env); the download itself only needs curl.
# shellcheck disable=SC1091
source "$ROOT/env.sh" >/dev/null || exit 1
set -eo pipefail
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
HUB="${HF_ENDPOINT:-https://huggingface.co}"
M="$ROOT/models"

want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
# setvars puts toolchain lib dirs (with an older libcurl) on LD_LIBRARY_PATH; the system curl
# works but warns "no version information available". Run it with the system libraries.
curl() { env -u LD_LIBRARY_PATH curl "$@"; }

# fetch_url URL OUT [SHA256]: download URL to OUT.
# Not `hf download`: its Xet transfer aborts on slow/unstable lines ("CAS Client Error ... error
# decoding response body") and discards partial data. Here partial data stays in OUT.part and
# `curl -C -` resumes it. The size comes from the server; the sha256 from the Hub's x-linked-etag
# header (LFS/Xet files) or the SHA256 argument.
fetch_url() {
  local url="$1" out="$2" sha="${3:-}" part="$2.part" name
  name="${out#"$M"/}"
  if [ -f "$out" ]; then echo "  have  $name"; return 0; fi
  mkdir -p "$(dirname "$out")"

  local head size etag
  head="$(curl -sSIL --connect-timeout 20 --retry 5 --retry-connrefused "$url" | tr -d '\r')" || true
  size="$(awk 'tolower($1)=="x-linked-size:"{l=$2} tolower($1)=="content-length:"{c=$2} END{print (l ? l : c)}' <<<"$head")"
  etag="$(awk 'tolower($1)=="x-linked-etag:"{gsub(/"/, "", $2); print $2}' <<<"$head")"
  [[ "$etag" =~ ^[0-9a-f]{64}$ ]] && sha="$etag"
  if ! grep -qE '^HTTP/[0-9.]+ 200' <<<"$head" || [ -z "$size" ]; then
    echo "cannot get $url (offline? wrong file name?)" >&2
    return 1
  fi
  echo "  get   $name ($(numfmt --to=iec --suffix=B "$size"))"

  local have stalls=0 rc
  while :; do
    have=$(stat -c %s "$part" 2>/dev/null || echo 0)
    [ "$have" -ge "$size" ] && break
    # --speed-limit/--speed-time: treat a connection stuck below 1 kB/s for a minute as dropped.
    curl -fL -C - --connect-timeout 20 --speed-limit 1024 --speed-time 60 -o "$part" "$url" && break
    rc=$?
    [ "$rc" = 33 ] && rm -f "$part"  # server refused to resume: start this file over
    if [ "$(stat -c %s "$part" 2>/dev/null || echo 0)" -gt "$have" ]; then stalls=0; else stalls=$((stalls + 1)); fi
    if [ "$stalls" -ge 10 ]; then
      echo "  no progress after 10 attempts; run the script again later to resume $name" >&2
      return 1
    fi
    echo "  connection dropped (curl exit $rc), resuming in 10 s..." >&2
    sleep 10
  done

  have=$(stat -c %s "$part")
  if [ "$have" != "$size" ]; then
    echo "  $name: got $have bytes, expected $size; deleted, run the script again" >&2
    rm -f "$part"
    return 1
  fi
  if [ -n "$sha" ]; then
    echo "  checking sha256..."
    if [ "$(sha256sum "$part" | cut -d' ' -f1)" != "$sha" ]; then
      echo "  $name: sha256 mismatch; deleted, run the script again" >&2
      rm -f "$part"
      return 1
    fi
  fi
  mv "$part" "$out"
}

# fetch REPO FILE DIR: a file from a Hugging Face model repo -> DIR/FILE
fetch() { fetch_url "$HUB/$1/resolve/main/$2" "$3/$2"; }

whisper_files() {  # whisper_files REPO DIR
  local f
  for f in config.json preprocessor_config.json tokenizer.json vocabulary.json model.bin; do
    fetch "$1" "$f" "$2"
  done
}

if want whisper; then
  step "ASR: faster-whisper large-v3-turbo (CTranslate2 fp16, 1.6 GB)"
  whisper_files mobiuslabsgmbh/faster-whisper-large-v3-turbo "$M/whisper/faster-whisper-large-v3-turbo"
  case "$ASR" in
    "") ;;
    large-v3)
      step "ASR (optional): faster-whisper large-v3 (CTranslate2 fp16, 3.1 GB)"
      whisper_files Systran/faster-whisper-large-v3 "$M/whisper/faster-whisper-large-v3"
      echo
      echo "To use it, run with --set asr.model=models/whisper/faster-whisper-large-v3 --set asr.compute_type=int8_float16"
      echo "(int8_float16 keeps it inside 8 GB next to the LLM; see README)" ;;
    *) echo "unknown --asr '$ASR' (large-v3)" >&2; exit 2 ;;
  esac
fi

if want llm; then
  case "$LLM" in
    translategemma-q6) REPO=mradermacher/translategemma-4b-it-GGUF; FILE=translategemma-4b-it.Q6_K.gguf; STYLE=translategemma ;;
    translategemma-q4) REPO=mradermacher/translategemma-4b-it-GGUF; FILE=translategemma-4b-it.Q4_K_M.gguf; STYLE=translategemma ;;
    translategemma-q8) REPO=mradermacher/translategemma-4b-it-GGUF; FILE=translategemma-4b-it.Q8_0.gguf; STYLE=translategemma ;;
    qwen3.5-4b)        REPO=unsloth/Qwen3.5-4B-GGUF;               FILE=Qwen3.5-4B-Q5_K_M.gguf;         STYLE=chat ;;
    gemma4-e4b)        REPO=ggml-org/gemma-4-E4B-it-GGUF;          FILE=gemma-4-E4B-it-Q4_0.gguf;       STYLE=chat ;;
    *) echo "unknown --llm '$LLM' (translategemma-q6|translategemma-q4|translategemma-q8|qwen3.5-4b|gemma4-e4b)" >&2; exit 2 ;;
  esac
  step "LLM: $REPO / $FILE"
  fetch "$REPO" "$FILE" "$M/llm"
  if [ "$FILE" != "translategemma-4b-it.Q6_K.gguf" ]; then
    echo
    echo "Not the default model. Set in config.toml [translation]:"
    echo "  model = \"models/llm/$FILE\""
    echo "  prompt_style = \"$STYLE\""
    echo "or run with:  --set translation.model=\"models/llm/$FILE\" --set translation.prompt_style=$STYLE"
  fi
fi

if want voices; then
  step "TTS: Piper voices (3 Italian, 4 English)"
  VOICES=(
    it/it_IT/paola/medium/it_IT-paola-medium.onnx          # Italian, female (default)
    it/it_IT/serena/high/it_IT-serena-high.onnx            # Italian, female
    it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx      # Italian, male (low quality)
    en/en_US/lessac/high/en_US-lessac-high.onnx            # US English, female (default)
    en/en_US/ryan/high/en_US-ryan-high.onnx                # US English, male
    en/en_GB/cori/high/en_GB-cori-high.onnx                # British English, female
    en/en_GB/alan/medium/en_GB-alan-medium.onnx            # British English, male
  )
  for v in "${VOICES[@]}"; do
    fetch rhasspy/piper-voices "$v.json" "$M/piper"
    fetch rhasspy/piper-voices "$v" "$M/piper"
  done
fi

if want kokoro; then
  # Kokoro-82M (Apache-2.0): second TTS engine with more natural English voices and an Italian
  # male + female voice. ONNX export from the kokoro-onnx project (sha256 from its release page).
  step "TTS: Kokoro-82M (ONNX model + voice pack, 354 MB)"
  KR=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1
  fetch_url "$KR/kokoro-v1.0.onnx" "$M/kokoro/kokoro-v1.0.onnx" \
    beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a
  fetch_url "$KR/voices-v1.0.bin" "$M/kokoro/voices-v1.0.bin" \
    bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d
fi

step "done. Verify with: ./run.sh check && ./run.sh selftest"
