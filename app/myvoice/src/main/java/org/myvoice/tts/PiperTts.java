package org.myvoice.tts;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.myvoice.accel.Backend;
import org.myvoice.accel.OrtSessions;
import org.myvoice.audio.AudioClip;
import org.myvoice.core.MiniJson;

/**
 * Piper voices (VITS, ONNX) with espeak-ng phonemes: a port of piper's own synthesis
 * (phonemize -> phoneme ids with "^" "_" "$" -> model -> audio), one sentence at a time.
 */
public final class PiperTts implements Tts {
    private final Map<String, Voice> voices = new HashMap<>();   // language -> voice
    private final Backend backend;
    private final int threads;
    private final int sentenceSilenceMs;

    /** @param voiceFiles language -> voice .onnx (its .onnx.json next to it) */
    public PiperTts(Map<String, File> voiceFiles, Backend backend, int threads, int sentenceSilenceMs) throws Exception {
        this.backend = backend;
        this.threads = threads;
        this.sentenceSilenceMs = sentenceSilenceMs;
        try {
            for (Map.Entry<String, File> e : voiceFiles.entrySet()) {
                voices.put(e.getKey(), new Voice(e.getValue(), backend, threads));
            }
        } catch (Exception ex) {
            close();
            throw ex;
        }
    }

    @Override
    public AudioClip synthesize(String text, String lang) throws Exception {
        Voice v = voices.get(lang);
        if (v == null) {
            throw new IllegalStateException("no Piper voice for " + lang);
        }
        if (text.codePoints().noneMatch(Character::isLetterOrDigit)) {
            return AudioClip.empty(v.sampleRate);
        }
        List<AudioClip> parts = new ArrayList<>();
        for (List<String> sentence : EspeakPhonemizer.phonemize(v.espeakVoice, text)) {
            parts.add(v.speak(v.ids(sentence)));
        }
        return AudioClip.concat(parts, sentenceSilenceMs / 1000.0);
    }

    @Override
    public String describe() {
        return "Piper · " + backend.label + (backend == Backend.CPU || backend == Backend.XNNPACK ? " ×" + threads : "");
    }

    @Override
    public void close() {
        for (Voice v : voices.values()) {
            v.close();
        }
        voices.clear();
    }

    /** The voice config's phoneme_id_map. */
    public static Map<String, long[]> readIdMap(Map<?, ?> voiceConfig) {
        Map<String, long[]> idMap = new HashMap<>();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) voiceConfig.get("phoneme_id_map")).entrySet()) {
            List<?> l = (List<?>) e.getValue();
            long[] ids = new long[l.size()];
            for (int i = 0; i < ids.length; i++) {
                ids[i] = ((Number) l.get(i)).longValue();
            }
            idMap.put((String) e.getKey(), ids);
        }
        return idMap;
    }

    /** Piper's phonemes_to_ids: ^ _ (phoneme _)* $, skipping phonemes the voice doesn't know. */
    public static long[] phonemeIds(Map<String, long[]> idMap, List<String> phonemes) {
        List<Long> out = new ArrayList<>();
        add(idMap, out, "^");
        add(idMap, out, "_");
        for (String p : phonemes) {
            if (idMap.containsKey(p)) {
                add(idMap, out, p);
                add(idMap, out, "_");
            }
        }
        add(idMap, out, "$");
        long[] a = new long[out.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = out.get(i);
        }
        return a;
    }

    private static void add(Map<String, long[]> idMap, List<Long> out, String key) {
        for (long id : idMap.get(key)) {
            out.add(id);
        }
    }

    static final class Voice implements AutoCloseable {
        private final OrtEnvironment env = OrtSessions.env();
        private final OrtSession session;
        final int sampleRate;
        final String espeakVoice;
        private final Map<String, long[]> idMap = new HashMap<>();
        private final float noiseScale;
        private final float lengthScale;
        private final float noiseW;

        Voice(File onnx, Backend backend, int threads) throws Exception {
            Map<?, ?> cfg;
            try (InputStream in = new FileInputStream(onnx.getPath() + ".json")) {
                cfg = (Map<?, ?>) new MiniJson(new InputStreamReader(in, StandardCharsets.UTF_8)).parse();
            }
            sampleRate = ((Number) ((Map<?, ?>) cfg.get("audio")).get("sample_rate")).intValue();
            espeakVoice = (String) ((Map<?, ?>) cfg.get("espeak")).get("voice");
            Map<?, ?> inference = (Map<?, ?>) cfg.get("inference");
            noiseScale = ((Number) inference.get("noise_scale")).floatValue();
            lengthScale = ((Number) inference.get("length_scale")).floatValue();
            noiseW = ((Number) inference.get("noise_w")).floatValue();
            idMap.putAll(readIdMap(cfg));
            session = OrtSessions.open(onnx.getPath(), backend, threads);
        }

        long[] ids(List<String> phonemes) {
            return phonemeIds(idMap, phonemes);
        }

        AudioClip speak(long[] ids) throws OrtException {
            try (OnnxTensor input = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), new long[] {1, ids.length});
                 OnnxTensor lengths = OnnxTensor.createTensor(env, LongBuffer.wrap(new long[] {ids.length}), new long[] {1});
                 OnnxTensor scales = OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[] {noiseScale, lengthScale, noiseW}), new long[] {3});
                 OrtSession.Result r = session.run(Map.of("input", input, "input_lengths", lengths, "scales", scales))) {
                FloatBuffer fb = ((OnnxTensor) r.get(0)).getFloatBuffer();
                float[] audio = new float[fb.remaining()];
                fb.get(audio);
                for (int i = 0; i < audio.length; i++) {
                    audio[i] = Math.max(-1f, Math.min(1f, audio[i]));
                }
                return new AudioClip(audio, sampleRate);
            }
        }

        @Override
        public void close() {
            try {
                session.close();
            } catch (OrtException ignored) {
            }
        }
    }
}
