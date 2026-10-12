package org.myvoice.asr;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.myvoice.accel.Backend;
import org.myvoice.nativebridge.NativeLib;

/**
 * Whisper through whisper.cpp (ggml: CPU, Vulkan or OpenCL).
 *
 * <p>Language detection follows the laptop: decode once with "auto"; the detection
 * probabilities come back from the same run, and if Whisper picked a language outside the pair
 * (short Italian clips sometimes come back as Spanish), decode again forced to the better of
 * the two (asr.py).
 */
public final class WhisperCppAsr implements Asr {
    private final long ctx;
    private final String name;
    private final Backend backend;
    private final String deviceLabel;
    private final int threads;
    private final String initialPrompt;
    private final boolean shortAudioCtx;

    /**
     * @param device       ggml device name for GPU backends ("" = CPU)
     * @param shortAudioCtx shrink Whisper's 30 s window to the clip length (faster on slow phones)
     */
    public WhisperCppAsr(File model, Backend backend, String device, String deviceLabel, int threads,
                         String initialPrompt, boolean shortAudioCtx) {
        this.ctx = NativeLib.whisperInit(model.getPath(), backend == Backend.CPU ? "" : device, true);
        if (ctx == 0) {
            throw new IllegalStateException("whisper.cpp could not load " + model.getName() + " on " + backend.label);
        }
        this.name = model.getName().replace("ggml-", "").replace(".bin", "");
        this.backend = backend;
        this.deviceLabel = deviceLabel;
        this.threads = threads;
        this.initialPrompt = initialPrompt == null ? "" : initialPrompt;
        this.shortAudioCtx = shortAudioCtx;
    }

    /**
     * Whisper's encoder window in frames (1500 = 30 s), shortened for short clips. Measured with
     * whisper.cpp: a window under ~2.3x the clip makes Whisper repeat itself (laptop), and on the
     * Moto G15 only multiples of 256 hit the fast code path (4 s clip: 512 frames 1.1 s, 514 frames
     * 16 s, full window 3.5 s). So: 2.5x the clip, rounded up to a multiple of 256.
     */
    static int audioCtxFor(int samples) {
        int frames = (int) Math.ceil(samples / 16000.0 * 50);
        int needed = (int) Math.ceil(2.5 * frames);
        int rounded = (needed + 255) / 256 * 256;
        return Math.min(1500, Math.max(512, rounded));
    }

    /** Whisper sometimes writes the same sentence twice in a row (seen on the phone): keep one. */
    static String dropRepeatedSentences(String text) {
        StringBuilder out = new StringBuilder();
        String last = null;
        for (String part : text.strip().split("(?<=[.!?…])\\s+")) {
            String key = part.toLowerCase(java.util.Locale.ROOT).replaceAll("[^\\p{L}\\p{N}]", "");
            if (key.isEmpty() || key.equals(last)) {
                continue;
            }
            out.append(out.length() > 0 ? " " : "").append(part);
            last = key;
        }
        return out.toString();
    }

    /** More words than anyone can say in that time: Whisper got stuck repeating itself. */
    static boolean looksRepeated(String text, int samples) {
        double seconds = samples / 16000.0;
        int words = text.isBlank() ? 0 : text.strip().split("\\s+").length;
        return words > 6 * seconds + 4;
    }

    private float[] lastAudio = new float[0];
    private int lastSamples;

    @Override
    public synchronized Transcript transcribe(float[] audio, List<String> allowed, String forced) {
        lastAudio = audio;
        lastSamples = audio.length;
        int actx = shortAudioCtx ? audioCtxFor(audio.length) : 0;
        if (forced != null) {
            return finish(NativeLib.whisperTranscribe(ctx, audio, forced, threads, initialPrompt, actx, null), 1.0f);
        }
        float[] probs = new float[NativeLib.whisperLangCount()];
        String out = NativeLib.whisperTranscribe(ctx, audio, "", threads, initialPrompt, actx, probs);
        if (out == null) {
            return null;
        }
        String detected = out.substring(0, out.indexOf('\t'));
        Map<String, Float> byLang = new HashMap<>();
        for (String code : allowed) {
            int id = NativeLib.whisperLangId(code);
            if (id >= 0 && id < probs.length) {
                byLang.put(code, probs[id]);
            }
        }
        LanguageChooser.Choice c = LanguageChooser.choose(byLang, detected, 0f, allowed);
        if (!c.language().equals(detected)) {
            out = NativeLib.whisperTranscribe(ctx, audio, c.language(), threads, initialPrompt, actx, null);
        }
        return finish(out, c.confidence(), c.language());
    }

    private Transcript finish(String out, float prob) {
        return out == null ? null : finish(out, prob, out.substring(0, out.indexOf('\t')));
    }

    private Transcript finish(String out, float prob, String lang) {
        if (out == null) {
            return null;
        }
        if (shortAudioCtx && looksRepeated(out.substring(out.indexOf('\t') + 1), lastSamples)) {
            // retry once with Whisper's full 30 s window
            String again = NativeLib.whisperTranscribe(ctx, lastAudio, lang, threads, initialPrompt, 0, null);
            if (again != null) {
                out = again;
            }
        }
        String text = dropRepeatedSentences(out.substring(out.indexOf('\t') + 1));
        if (text.isEmpty() || HallucinationFilter.isHallucination(text)) {
            return null;
        }
        return new Transcript(text, lang, prob);
    }

    @Override
    public String describe() {
        return "Whisper " + name + " · " + (backend == Backend.CPU ? "CPU ×" + threads : deviceLabel);
    }

    @Override
    public synchronized void close() {
        NativeLib.whisperFree(ctx);
    }
}
