#!/usr/bin/env bash
# Download the model files into ./models (needs internet + the myvoice env for the `hf` CLI).
#
#   bash scripts/download_models.sh                      # defaults (~4.9 GB): whisper turbo + TranslateGemma-4B Q6_K + 2 voices
#   bash scripts/download_models.sh --llm translategemma-q4   # smaller LLM (2.5 GB)
#   bash scripts/download_models.sh --llm qwen3.5-4b          # alternative general LLM (needs prompt_style="chat")
#   bash scripts/download_models.sh --extra-voices            # + more Italian/English voices
#   bash scripts/download_models.sh --only llm|whisper|voices
#
# Every file is pinned to an explicit repo + filename so you can also download by hand
# (see to-download.md). Downloads use plain HTTPS via curl and are resumable: if the connection
# drops, the script retries, and if it is killed, running it again continues where it stopped.
# Finished files are checked against the size and sha256 published by the Hub, then skipped on
# later runs.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LLM="translategemma-q6"
EXTRA_VOICES=0
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --llm) LLM="$2"; shift 2 ;;
    --extra-voices) EXTRA_VOICES=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
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

# fetch REPO FILE DIR: download REPO's FILE to DIR/FILE.
# Not `hf download`: its Xet transfer aborts on slow/unstable lines ("CAS Client Error ... error
# decoding response body") and discards partial data. Here partial data stays in DIR/FILE.part and
# `curl -C -` resumes it.
fetch() {
  local repo="$1" file="$2" out="$3/$2"
  local url="$HUB/$repo/resolve/main/$file" part="$3/$2.part"
  if [ -f "$out" ]; then echo "  have  $file"; return 0; fi
  mkdir -p "$(dirname "$out")"

  # Expected size and sha256 (x-linked-* headers; the etag is a sha256 only for LFS/Xet files).
  local head size sha
  head="$(curl -sSIL --connect-timeout 20 --retry 5 --retry-connrefused "$url" | tr -d '\r')" || true
  size="$(awk 'tolower($1)=="x-linked-size:"{l=$2} tolower($1)=="content-length:"{c=$2} END{print (l ? l : c)}' <<<"$head")"
  sha="$(awk 'tolower($1)=="x-linked-etag:"{gsub(/"/, "", $2); print $2}' <<<"$head")"
  [[ "$sha" =~ ^[0-9a-f]{64}$ ]] || sha=""
  if ! grep -qE '^HTTP/[0-9.]+ 200' <<<"$head" || [ -z "$size" ]; then
    echo "cannot get $url (offline? wrong file name?)" >&2
    return 1
  fi
  echo "  get   $file ($(numfmt --to=iec --suffix=B "$size"))"

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
      echo "  no progress after 10 attempts; run the script again later to resume $file" >&2
      return 1
    fi
    echo "  connection dropped (curl exit $rc), resuming in 10 s..." >&2
    sleep 10
  done

  have=$(stat -c %s "$part")
  if [ "$have" != "$size" ]; then
    echo "  $file: got $have bytes, expected $size; deleted, run the script again" >&2
    rm -f "$part"
    return 1
  fi
  if [ -n "$sha" ]; then
    echo "  checking sha256..."
    if [ "$(sha256sum "$part" | cut -d' ' -f1)" != "$sha" ]; then
      echo "  $file: sha256 mismatch; deleted, run the script again" >&2
      rm -f "$part"
      return 1
    fi
  fi
  mv "$part" "$out"
}

if want whisper; then
  step "ASR: faster-whisper large-v3-turbo (CTranslate2 fp16, 1.6 GB)"
  for f in config.json preprocessor_config.json tokenizer.json vocabulary.json model.bin; do
    fetch mobiuslabsgmbh/faster-whisper-large-v3-turbo "$f" "$M/whisper/faster-whisper-large-v3-turbo"
  done
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
  step "TTS: Piper voices (it_IT-paola-medium, en_US-lessac-high)"
  VOICES=(
    it/it_IT/paola/medium/it_IT-paola-medium.onnx
    en/en_US/lessac/high/en_US-lessac-high.onnx
  )
  if [ "$EXTRA_VOICES" = 1 ]; then
    VOICES+=(
      it/it_IT/serena/high/it_IT-serena-high.onnx
      it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx
      en/en_US/ryan/high/en_US-ryan-high.onnx
      en/en_GB/cori/high/en_GB-cori-high.onnx
    )
  fi
  for v in "${VOICES[@]}"; do
    fetch rhasspy/piper-voices "$v.json" "$M/piper"
    fetch rhasspy/piper-voices "$v" "$M/piper"
  done
fi

step "done. Verify with: ./run.sh check && ./run.sh selftest"
