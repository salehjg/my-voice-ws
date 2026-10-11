# myvoice: talk with Italians, offline

An Italian speaks; you read (and hear) it in English. You answer in English; they hear it in
Italian. Everything runs on this laptop (RTX 2000 Ada, 8 GB) with no internet.

```bash
cd /workspace/02_ws/my-voice-ws
./run.sh
```

That's it. Tip: add `alias myvoice=/workspace/02_ws/my-voice-ws/run.sh` to `~/.zshrc`, then just
type `myvoice` from anywhere.

## The conversation app

Keep pressing **SPACE** to move through the cycle:

```
 SPACE ──► listen to Italian ──► English text (+ English voice) ──► "Your turn"
   ▲                                                                    │ SPACE
   │                                                                    ▼
   └──── Italian voice speaks your answer ◄── ENTER (check) ◄── you speak English
```

1. **SPACE**: the app listens to the Italian speaker. A red dot and a level bar show it hears
   them. It stops by itself after they pause (1 s). You can also press SPACE to stop it yourself.
2. The Italian text and its **English translation** appear, and an English voice reads it.
3. **SPACE**: now *you* answer in English. Your turn ends after a 1.5 s pause, or press SPACE.
4. The app shows **what it heard from you**. Press **ENTER** if it's right. Press **E** to fix a
   word (or type the whole thing), **R** to say it again, or **ESC** to cancel.
5. Your answer is translated and **spoken in Italian**. Back to step 1.

Every step can be skipped or done differently:

| Key | What it does |
|---|---|
| **SPACE** | next step (listen to them / record your answer / confirm) |
| **E** | answer in English by voice right away (skip listening to them) |
| **T** | type your answer in English instead of speaking (translated and spoken in Italian) |
| **I** | type something in Italian (e.g. from a sign or a message): shown in English |
| **L** | listen to them again (when they keep talking) |
| **F** | translate an audio file: drag a voice message into the terminal, press Enter |
| **R** | replay the last thing that was spoken |
| **A** | English voice on/off (off = you only read the translation) |
| **C** | check-my-speech step on/off (off = your answer is translated right away) |
| **V** | change voices (each press plays a sample) |
| **S** / **ESC** | skip / cancel the current step |
| **H** | show all keys |
| **Q** | quit |

While recording, **SPACE** or **ENTER** finishes and **ESC** cancels. While a voice is speaking,
**any key** stops it. Your choices (voices, A, C) are remembered in `config.local.toml`.

## Other ways to use it

```bash
./run.sh file messaggio.ogg --play         # an Italian voice message / recording -> English text + voice
./run.sh file intervista.m4a -o out.wav    # ...or save the spoken translation (+ a .jsonl transcript)
./run.sh text "Dov'è la farmacia?"         # typed text, either language (add --speak to hear it)
./run.sh live --headphones                 # hands-free: translates everything it hears, both ways
./run.sh voices                            # list voices; --play to hear them; --use ID to pick one
./run.sh --help                            # everything else (check, devices, selftest)
```

`file` reads wav, mp3, ogg/opus (WhatsApp voice notes), m4a and flac.

## Voices

| | Italian (what they hear) | English (what you hear) |
|---|---|---|
| **default** | `kokoro:if_sara` (female) | `kokoro:af_heart` (female, US) |
| other Kokoro | `kokoro:im_nicola` (male) | `af_bella`, `am_michael`, `am_fenrir` (US); `bf_emma`, `bm_george` (UK) |
| Piper | `it_IT-paola-medium`, `it_IT-serena-high` (female), `it_IT-riccardo-x_low` (male, robotic) | `en_US-lessac-high`, `en_US-ryan-high`, `en_GB-cori-high`, `en_GB-alan-medium` |

To change voices, press **V** in the app, or run `./run.sh voices --use kokoro:im_nicola`. Kokoro
has about 20 more English voices: `./run.sh voices --all`.

Why Kokoro is the default: each voice spoke test sentences, and Whisper wrote down what it heard.
Kokoro's Italian came back word-perfect. Piper's Paola was heard as "in *dettaglio*" instead of
"in *ritardo*".

## Getting your English right

Whisper is told which language to expect at each step, so it never mistakes your English for
Italian. Beyond that:

- **Check step (on by default):** you see what was understood before it's translated. **E** lets
  you correct it, **R** lets you repeat it.
- **Level bar:** if the bar barely moves while you talk, the mic is too far away or muted. If
  the app didn't notice you speaking, **SPACE** still keeps what you said.
- **Pauses:** you can pause for up to 1.5 s while thinking. To allow longer pauses:
  `./run.sh talk --set talk.my_end_silence_ms=2500`.
- **Names and jargon:** put them in `asr.initial_prompt` in `config.toml`, e.g.
  `initial_prompt = "Saleh, Bologna, Unipol, codice fiscale"`.
- **A headset mic** helps more than any setting.
- A bigger speech model (Whisper large-v3, already downloaded) did **not** do better. On 192
  accented-English test sentences it scored 4.3–4.5 % word errors against the default's 4.6 %,
  and was 50 % slower. An LLM "fixing" your transcript can invent things you didn't say, so it
  isn't used. The check step does that job safely.

## Settings

Everything is in [`config.toml`](config.toml) (commented). Any value can be changed for one run:
`./run.sh talk --set section.key=value`. The useful ones:

| Setting | Default | Change it when |
|---|---|---|
| `talk.their_end_silence_ms` | 1000 | they get cut off mid-sentence (raise) / it waits too long (lower) |
| `talk.my_end_silence_ms` | 1500 | the same, for you |
| `talk.speak_to_me` | true | you only want to read (key **A**) |
| `talk.confirm_my_speech` | true | you trust it and want speed (key **C**) |
| `vad.threshold` | 0.5 | noisy room: 0.6–0.7 |
| `tts.length_scale` | 1.0 | 1.15 = slower, clearer voices |
| `audio.input_device` / `output_device` | system default | `./run.sh devices`, then `--input-device "USB"` |
| `asr.initial_prompt` | "" | names or words it keeps getting wrong |

`config.local.toml` holds what you picked in the app. Delete it to go back to the defaults.

## How it works

```
 mic ─► Silero VAD ─► Whisper large-v3-turbo ─► TranslateGemma 4B ─► Kokoro / Piper ─► speaker
         (CPU)          (GPU, fp16)               (GPU, llama.cpp)     (CPU)
```

| Part | Model | Notes |
|---|---|---|
| Voice detection | Silero VAD | finds where speech starts and ends |
| Speech → text | Whisper large-v3-turbo (faster-whisper) | language fixed per step in the app; auto-detected in `live`/`file` |
| Translation | TranslateGemma 4B Q6_K (Google, Jan 2026) | one model for both directions, greedy decoding |
| Text → speech | Kokoro-82M / Piper | on the CPU, sentence by sentence (the voice starts while the rest is prepared) |

Measured on this laptop: about 6 GB of 8 GB VRAM. After someone stops talking, you get the
pause detection (1 s), then about 0.3–0.5 s to transcribe and 0.2–0.5 s to translate. The text
appears then, and the voice starts about 0.5–1 s later.

## First-time setup

On this laptop everything is installed already. For a new machine or a rebuild, see
[`to-download.md`](to-download.md): run `bash scripts/setup_env.sh`, then
`bash scripts/download_models.sh`, then `./run.sh check` and `./run.sh selftest`. Every script
sources `/home/saleh/00_software/dir_install/setvars.all.sh` first (through `env.sh`), and
re-running them only fetches what's missing.

## Troubleshooting

| Problem | Fix |
|---|---|
| "nothing recorded" though you spoke | watch the level bar; check the mic with `./run.sh devices` and `--input-device`; press SPACE to stop by hand |
| It cuts people off | raise `talk.their_end_silence_ms` / `talk.my_end_silence_ms` |
| Wrong words in your English | press **E** at the check step; add names to `asr.initial_prompt`; use a headset |
| No sound | `./run.sh devices`, then `./run.sh talk --output-device <index or name>` |
| `missing model` / something fails at start | `./run.sh check` names what's missing; `bash scripts/download_models.sh` fetches only that |
| `check`: llama-cpp-python *crashed (illegal instruction)* | `bash scripts/setup_env.sh` (rebuilds it for this CPU, which has no AVX-512) |
| CUDA out of memory | close other GPU programs, or `--set asr.compute_type=int8_float16` |
| Weird voice settings stuck | delete `config.local.toml` |

## For developers

```
run.sh / env.sh            entry point / environment (setvars + conda env + CUDA libs + offline mode)
config.toml                settings (+ config.local.toml: choices saved by the app)
scripts/                   setup_env.sh, download_models.sh, patch_cuda_glibc.sh
src/myvoice/cli.py         commands (Typer)            talk.py      the conversation app (Rich)
src/myvoice/pipeline.py    Engine, live mode, files    asr.py       Whisper wrapper
src/myvoice/translate.py   prompts + llama.cpp         tts.py       Kokoro + Piper voices
src/myvoice/vad.py         voice detection             audio.py     mic, playback, file I/O
tests/                     unit tests with fake models: source env.sh && python -m pytest -q tests
```

For testing without a microphone, `audio.input_device = "file:a.wav,b.wav"` plays files into the
app as if they were spoken: each recording uses the next file.
