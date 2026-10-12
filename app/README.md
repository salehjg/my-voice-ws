# myvoice for Android

The laptop translator on a phone: Italian ⇄ English, offline, in **auto mode** (translates
everything it hears) or **manual mode** (buttons for each step). Written in Java, with the models
running through whisper.cpp, llama.cpp, ONNX Runtime and espeak-ng.

## Build and install

Android Studio (`android-studio`) can open this `app/` folder directly. From a terminal:

```bash
cd /workspace/02_ws/my-voice-ws
git submodule update --init          # native sources in app/third_party (pinned versions)
cd app
export JAVA_HOME=/opt/android-studio/jbr
./gradlew assembleDebug              # first build compiles the native code (~10 min)
adb install -r myvoice/build/outputs/apk/debug/myvoice-debug.apk
```

The phone needs USB debugging (Settings → About phone → tap "Build number" 7 times → Developer
options → USB debugging).

## First start

1. **Models.** The setup screen picks the laptop's models by default:
   - Whisper large-v3-turbo for speech recognition;
   - TranslateGemma 4B Q6_K for translation, which is exactly the laptop's file.

   Each choice shows its size, whether it fits in memory, and a rough time per sentence on the
   phone. On a Moto G15 the laptop models take tens of seconds per sentence; "Fast" (Whisper base +
   Opus-MT) takes a few seconds.

   Files you already have on the laptop can be copied instead of downloaded. Open the app once
   first: it creates its folders, and Android only lets the app read files pushed into folders it
   made itself.

   ```bash
   adb push models/llm/translategemma-4b-it.Q6_K.gguf /sdcard/Android/data/org.myvoice/files/models/llm/
   ```

   Then tap **Download / check**: copied files are checked against their sha256, not downloaded
   again. Downloads resume after a dropped connection.
2. **Voices.** The phone's own offline voices are used first; male ones are picked where they are
   known. A Piper voice per language is the fallback. If a language is missing, "Install the phone's
   voice data" opens Android's voice download.
3. **Optimise for this phone.** Each model is tried on every accelerator the phone has:
   - Vulkan and OpenCL GPUs (through ggml);
   - NNAPI (GPU/NPU) and QNN (Qualcomm NPU), through ONNX Runtime;
   - the CPU, with a few thread counts.

   The app keeps the fastest one that gives the same answer as the CPU. Settings → Acceleration
   shows every result:
   - ✓ used
   - ✓ slower (with both times)
   - ✗ failed or crashed (with the reason)
   - – unavailable

   Probing runs in a separate process, so a GPU driver that crashes can't take the app down.

## Speed on the Moto G15 (measured)

Helio G81, 2× Cortex-A75 + 6× Cortex-A55, Mali-G52 MC2, 7.5 GB. Times are for one ~4 s sentence.
The probe found:
- **Vulkan works but is far slower**: Whisper base took 11.8 s on Vulkan against 1.1 s on the CPU.
- **OpenCL is unavailable**: ggml's OpenCL backend supports Adreno GPUs only.
- **NNAPI** reports no GPU/NPU device on this chip.
- **XNNPACK** works but is slower than plain ONNX Runtime CPU.

So everything runs on the CPU, with 4 threads; 8 threads are slower because of the little cores.

| Model | Time on the CPU |
|---|---|
| Whisper base (with language detection) | 1.5 s |
| Whisper small | 4.9 s |
| Whisper large-v3-turbo (the laptop model) | 44 s |
| Opus-MT | 2–3.5 s |
| TranslateGemma 4B Q6_K (the laptop model) | 39 s |
| Piper voice | 0.5–1 s |

With the **Fast** preset, the text appears about 4–5 s after someone stops talking, and the voice
follows.

## Using it

- **Auto mode:** just talk; the mic pauses while a translation is spoken, unless headphones are
  connected. *Wrong language* redoes the last sentence as the other language.
- **Manual mode:** *Listen to them* → English text and voice → *My turn* → check what was heard →
  spoken in Italian.
- **The chips at the top** show what each step runs on, e.g. `Whisper base · CPU ×4`. If an
  accelerator fails in a conversation, the app switches to the next one and says so.
- **Background:** it keeps running in the background and with the screen off. The notification
  shows the last translation and has **Pause/Resume** and **Exit**.
- **Exit** (notification, ✕ on the main screen, or Settings) stops listening, frees the models and
  closes the app.

## Tests

```bash
./gradlew :myvoice:testDebugUnitTest          # conversation logic, segmenter, prompts, catalogue (JVM)
./gradlew connectedDebugAndroidTest           # on the phone: native libraries, espeak, Silero, models
# Java engines vs the laptop's Python tools, with real models (see the script's docstring):
python tools/make_test_vectors.py OPUS_DIR golden.json
MYVOICE_OPUS_DIR=OPUS_DIR MYVOICE_GOLDEN=golden.json ./gradlew -p tools/jvm-check test
# the native code (whisper.cpp, llama.cpp, espeak-ng) built for the laptop, CPU only:
cmake -G Ninja -S tools/native_host_test -B /tmp/mv-host && ninja -C /tmp/mv-host host_test
# ...or for the phone, to time models directly over adb (no app needed):
cmake -G Ninja -S tools/native_host_test -B /tmp/mv-phone -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
      -DCMAKE_TOOLCHAIN_FILE=$HOME/Android/Sdk/ndk/28.2.13676358/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-33 && ninja -C /tmp/mv-phone host_test
adb push /tmp/mv-phone/host_test ggml-base-q8_0.bin clip.wav /data/local/tmp/
adb shell "cd /data/local/tmp && ./host_test whisper ggml-base-q8_0.bin clip.wav auto 4 512"
```

**Emulator:** `./gradlew -Pemulator assembleDebug` adds x86_64 native code, so the app runs on
Android Studio's x86_64 emulator without ARM translation.

**Debug input.** Settings → *Test audio input* plays `.wav` files from
`/sdcard/Android/data/org.myvoice/files/test_audio/` as if they were spoken, like the laptop's
`--input-device file:...`. Transcripts go to `.../files/sessions/*.jsonl`, with the same fields as
the laptop's.

## Layout

```
myvoice/src/main/java/org/myvoice/
  core/       Conversation (both modes), Engine, settings, session log, fallbacks
  vad/ asr/ mt/ tts/ audio/   the engines and the phone's mic/speaker
  accel/      device facts, probing, backend choice
  models/     catalogue (assets/models.json), downloads, estimates
  service/    TranslatorService (microphone), DownloadService, ProbeService (:probe process)
  ui/         Setup, Main, Settings, Acceleration screens
myvoice/src/main/cpp/          JNI + plain C++ (also built for the laptop by tools/native_host_test)
third_party/                   llama.cpp, whisper.cpp, espeak-ng, Vulkan/SPIRV/OpenCL headers (submodules)
```

## Licences

whisper.cpp and llama.cpp are MIT; ONNX Runtime is MIT; Silero VAD is MIT; Opus-MT is CC-BY 4.0;
Piper voices have their own model cards; TranslateGemma and Gemma follow the Gemma Terms of Use.
**espeak-ng is GPL-3.0, so this app as a whole is GPL-3.0.** That is fine for personal use.
