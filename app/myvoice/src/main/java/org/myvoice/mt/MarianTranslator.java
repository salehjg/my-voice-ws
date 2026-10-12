package org.myvoice.mt;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
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
import org.myvoice.core.MiniJson;

/**
 * Opus-MT (Marian) translation with ONNX Runtime: one model per direction (it→en, en→it), each an
 * encoder plus a "merged" decoder with a key/value cache, decoded greedily.
 *
 * <p>Model folder layout (the Xenova exports): {@code config.json}, {@code tokenizer.json},
 * {@code onnx/encoder_model_quantized.onnx}, {@code onnx/decoder_model_merged_quantized.onnx}.
 */
public final class MarianTranslator implements Translator {
    private final Map<String, Model> models = new HashMap<>();
    private final Backend backend;
    private final int threads;

    /** @param dirs direction ("it-en") -> model folder */
    public MarianTranslator(Map<String, File> dirs, Backend backend, int threads) throws IOException, OrtException {
        this.backend = backend;
        this.threads = threads;
        try {
            for (Map.Entry<String, File> e : dirs.entrySet()) {
                models.put(e.getKey(), new Model(e.getValue(), backend, threads));
            }
        } catch (IOException | OrtException | RuntimeException ex) {
            close();
            throw ex;
        }
    }

    @Override
    public String translate(String text, String src, String tgt) throws OrtException {
        Model m = models.get(src + "-" + tgt);
        if (m == null) {
            throw new IllegalStateException("no Opus-MT model for " + src + " → " + tgt);
        }
        // Marian models tend to stop after the first sentence: translate one sentence at a time
        StringBuilder out = new StringBuilder();
        for (String part : SENTENCE_END.split(text.strip())) {
            if (!part.isBlank()) {
                out.append(out.length() > 0 ? " " : "").append(m.translate(part));
            }
        }
        return OutputCleaner.clean(out.toString(), text);
    }

    private static final java.util.regex.Pattern SENTENCE_END = java.util.regex.Pattern.compile("(?<=[.!?…])\\s+(?=\\S)");

    @Override
    public String describe() {
        return "Opus-MT · " + backend.label + (backend == Backend.CPU || backend == Backend.XNNPACK ? " ×" + threads : "");
    }

    @Override
    public void close() {
        for (Model m : models.values()) {
            m.close();
        }
        models.clear();
    }

    /** One direction. */
    static final class Model implements AutoCloseable {
        private final OrtEnvironment env = OrtSessions.env();
        private final UnigramTokenizer tok;
        private final OrtSession encoder;
        private final OrtSession decoder;
        private final int heads;
        private final int headDim;
        private final long startId;
        private final long eosId;
        private final long padId;
        private final List<String> pastNames = new ArrayList<>();

        Model(File dir, Backend backend, int threads) throws IOException, OrtException {
            Map<?, ?> cfg;
            try (InputStream in = new FileInputStream(new File(dir, "config.json"))) {
                cfg = (Map<?, ?>) new MiniJson(new InputStreamReader(in, StandardCharsets.UTF_8)).parse();
            }
            try (InputStream in = new FileInputStream(new File(dir, "tokenizer.json"))) {
                tok = UnigramTokenizer.fromJson(in);
            }
            int dModel = num(cfg, "d_model", 512);
            heads = num(cfg, "decoder_attention_heads", 8);
            headDim = dModel / heads;
            startId = num(cfg, "decoder_start_token_id", tok.padId);
            eosId = num(cfg, "eos_token_id", tok.eosId);
            padId = num(cfg, "pad_token_id", tok.padId);
            File onnx = new File(dir, "onnx");
            encoder = OrtSessions.open(new File(onnx, "encoder_model_quantized.onnx").getPath(), backend, threads);
            try {
                decoder = OrtSessions.open(new File(onnx, "decoder_model_merged_quantized.onnx").getPath(), backend, threads);
            } catch (OrtException e) {
                encoder.close();
                throw e;
            }
            for (String name : decoder.getInputNames()) {
                if (name.startsWith("past_key_values.")) {
                    pastNames.add(name);
                }
            }
        }

        private static int num(Map<?, ?> cfg, String key, int fallback) {
            Object v = cfg.get(key);
            return v instanceof Number n ? n.intValue() : fallback;
        }

        String translate(String text) throws OrtException {
            List<Integer> src = tok.encode(text);
            long[] ids = new long[src.size()];
            long[] mask = new long[src.size()];
            for (int i = 0; i < ids.length; i++) {
                ids[i] = src.get(i);
                mask[i] = 1;
            }
            long[] shape = {1, ids.length};
            int maxLen = Math.min(512, 2 * ids.length + 20);
            List<Integer> out = new ArrayList<>();
            try (OnnxTensor inIds = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape);
                 OnnxTensor inMask = OnnxTensor.createTensor(env, LongBuffer.wrap(mask), shape);
                 OrtSession.Result enc = encoder.run(Map.of("input_ids", inIds, "attention_mask", inMask))) {
                OnnxTensor hidden = (OnnxTensor) enc.get(0);
                OrtSession.Result prev = null;    // previous step: its decoder key/values
                OrtSession.Result first = null;   // first step: its encoder key/values (the cached branch
                                                  // only outputs placeholders for those)
                long last = startId;
                try {
                    for (int step = 0; step < maxLen; step++) {
                        Map<String, OnnxTensor> feed = new HashMap<>();
                        List<OnnxTensor> owned = new ArrayList<>();
                        OnnxTensor stepIds = OnnxTensor.createTensor(env, LongBuffer.wrap(new long[] {last}), new long[] {1, 1});
                        owned.add(stepIds);
                        feed.put("input_ids", stepIds);
                        feed.put("encoder_attention_mask", inMask);
                        feed.put("encoder_hidden_states", hidden);
                        boolean cached = prev != null;
                        if (decoder.getInputNames().contains("use_cache_branch")) {
                            OnnxTensor flag = OnnxTensor.createTensor(env, new boolean[] {cached});
                            owned.add(flag);
                            feed.put("use_cache_branch", flag);
                        }
                        for (String name : pastNames) {
                            OnnxTensor t;
                            if (cached) {
                                OrtSession.Result from = name.contains(".encoder.") ? first : prev;
                                t = (OnnxTensor) from.get(name.replace("past_key_values.", "present.")).orElseThrow();
                            } else {
                                t = OnnxTensor.createTensor(env, FloatBuffer.allocate(0), new long[] {1, heads, 0, headDim});
                                owned.add(t);
                            }
                            feed.put(name, t);
                        }
                        OrtSession.Result res = decoder.run(feed);
                        for (OnnxTensor t : owned) {
                            t.close();
                        }
                        if (prev != null && prev != first) {
                            prev.close();
                        }
                        if (first == null) {
                            first = res;
                        }
                        prev = res;
                        OnnxTensor logits = (OnnxTensor) res.get("logits").orElseThrow();
                        long[] ls = logits.getInfo().getShape();   // [1, seq, vocab]
                        int vocab = (int) ls[2];
                        FloatBuffer fb = logits.getFloatBuffer();
                        int base = (int) ((ls[1] - 1) * vocab);
                        int best = -1;
                        float bestV = Float.NEGATIVE_INFINITY;
                        for (int v = 0; v < vocab; v++) {
                            if (v == padId) {
                                continue;   // bad_words_ids: never produce <pad>
                            }
                            float x = fb.get(base + v);
                            if (x > bestV) {
                                bestV = x;
                                best = v;
                            }
                        }
                        if (best == eosId || best < 0) {
                            break;
                        }
                        out.add(best);
                        last = best;
                    }
                } finally {
                    if (prev != null && prev != first) {
                        prev.close();
                    }
                    if (first != null) {
                        first.close();
                    }
                }
            }
            return tok.decode(out);
        }

        @Override
        public void close() {
            try {
                encoder.close();
            } catch (OrtException ignored) {
            }
            try {
                decoder.close();
            } catch (OrtException ignored) {
            }
        }
    }
}
