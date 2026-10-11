# myvoice: talk with Italians, offline

Speak Italian or English; it is translated into the other language, shown on screen and spoken
aloud. Everything runs on this laptop (RTX 2000 Ada, 8 GB) with no internet.

```bash
cd /workspace/02_ws/my-voice-ws
./myvoice.py          # auto mode: just talk
./myvoice.py -m       # manual mode: you start and skip each step with keys
```

That's it. In the app, **TAB** switches between the two modes. Tip: add
`alias myvoice=/workspace/02_ws/my-voice-ws/myvoice.py` to `~/.zshrc`, then just type `myvoice`
(or `myvoice -m`) from anywhere.

## Auto mode (default)

The microphone is always on. Each sentence is recognised as Italian or English, translated into
the other language, shown, and spoken: Italian with the Italian voice, English with the English one.

```
 ● Listening ──► someone talks ──► pause ──► Italian → English (or English → Italian) ──► spoken
      ▲                                                                                    │
      └────────────────────────────────────────────────────────────────────────────────────┘
```

- **Wait for "● Listening"** before the next sentence. With speakers, the mic pauses while a
  translation is spoken, so the app doesn't translate its own voice. With headphones, start it
  with `./myvoice.py --headphones` and it keeps listening the whole time.
- A sentence ends after a 1 s pause.
- Very short phrases ("Sì", "OK") can be taken for the wrong language. If that happens, press
  **W** and the last sentence is redone as the other language. When the app is unsure, the panel
  says so.

| Key | What it does |
|---|---|
| **SPACE** | pause / resume listening |
| **TAB** | switch to manual mode |
| **W** | wrong language: redo the last sentence as the other language |
| **T** | type a sentence (either language) |
| **R** | replay the last translation |
| **A** | English voice on/off (off = translations into English are only shown) |
| **V** | change voices (each press plays a sample) |
| **H** | show all keys |
| **Q** | quit |

While a voice is speaking, **any key** stops it.

## Manual mode (`-m`)

For when you want control: the app listens only when you say so, and you can check what it
understood before your answer is translated. Keep pressing **SPACE** to move through the cycle:

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
| **TAB** | switch to auto mode |
| **H** | show all keys |
| **Q** | quit |

While recording, **SPACE** or **ENTER** finishes and **ESC** cancels. Your choices (voices, A, C)
are remembered in `config.local.toml`.

## Other ways to use it

```bash
./myvoice.py file messaggio.ogg --play       # an Italian voice message / recording -> English text + voice
./myvoice.py file intervista.m4a -o out.wav  # ...or save the spoken translation (+ a .jsonl transcript)
./myvoice.py text "Dov'è la farmacia?"       # typed text, either language (add --speak to hear it)
./myvoice.py voices                          # list voices; --play to hear them; --use ID to pick one
./myvoice.py -h                              # everything else (devices, check, selftest, setup, ...)
```

`file` reads wav, mp3, ogg/opus (WhatsApp voice notes), m4a and flac.

## Voices

| | Italian (what they hear) | English (what you hear) |
|---|---|---|
| **default** | `kokoro:im_nicola` (male) | `kokoro:am_michael` (male, US) |
| other Kokoro | `kokoro:if_sara` (female) | `af_heart`, `af_bella` (female, US); `am_fenrir` (male, US); `bf_emma`, `bm_george` (UK) |
| Piper | `it_IT-paola-medium`, `it_IT-serena-high` (female), `it_IT-riccardo-x_low` (male, robotic) | `en_US-lessac-high`, `en_US-ryan-high`, `en_GB-cori-high`, `en_GB-alan-medium` |

To change voices, press **V** in the app, or run `./myvoice.py voices --use kokoro:if_sara`.
Kokoro has about 20 more English voices: `./myvoice.py voices --all`.

Why Kokoro: each voice spoke test sentences, and Whisper wrote down what it heard. All seven
Kokoro voices tested (Italian and English, male and female) came back word-perfect apart from how
numbers are written. Piper's Paola was heard as "in *dettaglio*" instead of "in *ritardo*".

## Getting your English right

Whisper recognises the language of each sentence (auto mode) or is told which language to expect
(manual mode). Beyond that:

- **For answers that matter, use manual mode** (TAB). Its check step shows what was understood
  before it's translated: **E** lets you correct it, **R** lets you repeat it.
- **Level bar:** if the bar barely moves while you talk, the mic is too far away or muted.
- **Pauses:** in manual mode you can pause for up to 1.5 s while thinking (1 s in auto mode). To
  allow longer pauses: `./myvoice.py --set talk.auto_end_silence_ms=1500` (auto) or
  `./myvoice.py -m --set talk.my_end_silence_ms=2500` (manual).
- **Names and jargon:** put them in `asr.initial_prompt` in `config.toml`, e.g.
  `initial_prompt = "Saleh, Bologna, Unipol, codice fiscale"`.
- **A headset mic** helps more than any setting.
- A bigger speech model (Whisper large-v3, already downloaded) did **not** do better. On 192
  accented-English test sentences it scored 4.3–4.5 % word errors against the default's 4.6 %,
  and was 50 % slower. An LLM "fixing" your transcript can invent things you didn't say, so it
  isn't used. The check step does that job safely.

## Settings

Everything is in [`config.toml`](config.toml) (commented). Any value can be changed for one run:
`./myvoice.py --set section.key=value`. The useful ones:

| Setting | Default | Change it when |
|---|---|---|
| `talk.auto_end_silence_ms` | 1000 | auto mode cuts sentences in half (raise) / waits too long (lower) |
| `talk.their_end_silence_ms` | 1000 | manual mode: they get cut off mid-sentence (raise) / it waits too long (lower) |
| `talk.my_end_silence_ms` | 1500 | the same, for you |
| `talk.speak_to_me` | true | you only want to read the English (key **A**) |
| `talk.confirm_my_speech` | true | manual mode: you trust it and want speed (key **C**) |
| `audio.half_duplex` | true | false = keep listening while speaking; only with headphones (`--headphones`) |
| `vad.threshold` | 0.5 | noisy room: 0.6–0.7 |
| `tts.length_scale` | 1.0 | 1.15 = slower, clearer voices |
| `audio.input_device` / `output_device` | system default | `./myvoice.py devices`, then `--input-device "USB"` |
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
| Speech → text | Whisper large-v3-turbo (faster-whisper) | detects Italian vs English (auto mode, `file`); fixed per step in manual mode |
| Translation | TranslateGemma 4B Q6_K (Google, Jan 2026) | one model for both directions, greedy decoding |
| Text → speech | Kokoro-82M / Piper | on the CPU, sentence by sentence (the voice starts while the rest is prepared) |

Measured on this laptop: about 6 GB of 8 GB VRAM. After someone stops talking, you get the
pause detection (1 s), then about 0.3–0.5 s to transcribe and 0.3–0.5 s to translate. The text
appears then, and the voice starts about 0.5–1 s later.

`./myvoice.py` is plain Python and any `python3` can start it. It sources
`/home/saleh/00_software/dir_install/setvars.all.sh`, activates the `myvoice` conda env, puts the
CUDA libraries on the library path, turns on offline mode, and then runs the app inside that env.

## First-time setup

On this laptop everything is installed already. For a new machine or a rebuild, see
[`to-download.md`](to-download.md):

```bash
./myvoice.py setup       # conda env + packages (~5 GB; compiles llama.cpp for this CPU/GPU)
./myvoice.py download    # models into ./models (~5.8 GB; resumable, skips what's there)
./myvoice.py check       # every line should say ok
./myvoice.py selftest    # a voice speaks Italian + English through the whole chain
```

Re-running `setup` or `download` only fetches what's missing.

## Troubleshooting

| Problem | Fix |
|---|---|
| Auto mode doesn't react when you speak | watch the level bar; check the mic with `./myvoice.py devices` and `--input-device` |
| Auto mode translates its own voice | don't use `--headphones` with speakers; raise `audio.playback_tail_ms` (e.g. 600) |
| Wrong language detected (auto mode) | press **W**; for short replies use manual mode (TAB), where the language is fixed |
| It cuts people off | raise `talk.auto_end_silence_ms` (auto) or `talk.their_end_silence_ms` / `talk.my_end_silence_ms` (manual) |
| Wrong words in your English | manual mode: press **E** at the check step; add names to `asr.initial_prompt`; use a headset |
| No sound | `./myvoice.py devices`, then `./myvoice.py --output-device <index or name>` |
| `missing model` / something fails at start | `./myvoice.py check` names what's missing; `./myvoice.py download` fetches only that |
| `check`: llama-cpp-python *crashed (illegal instruction)* | `./myvoice.py setup` (rebuilds it for this CPU, which has no AVX-512) |
| CUDA out of memory | close other GPU programs, or `--set asr.compute_type=int8_float16` |
| Weird voice settings stuck | delete `config.local.toml` |

## For developers

```
myvoice.py                 entry point (any python3): loads setvars.all.sh + conda env, then runs the CLI
config.toml                settings (+ config.local.toml: choices saved by the app)
src/myvoice/launch.py      environment set-up (standard library only)
src/myvoice/install.py     `setup`: conda env, packages, llama.cpp build, CUDA/glibc header patch
src/myvoice/download.py    `download`: resumable, checksummed model downloads
src/myvoice/cli.py         commands (Typer)            talk.py      the conversation app, both modes (Rich)
src/myvoice/pipeline.py    Engine, file translation    asr.py       Whisper wrapper
src/myvoice/translate.py   prompts + llama.cpp         tts.py       Kokoro + Piper voices
src/myvoice/vad.py         voice detection             audio.py     mic, playback, file I/O
tests/                     unit tests with fake models: ./myvoice.py test
```

For testing without a microphone, `--input-device "file:a.wav,b.wav"` plays files into the app as
if they were spoken. Each time the mic starts, it uses the next file.
