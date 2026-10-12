package org.myvoice;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.junit.Test;
import org.myvoice.accel.Backend;
import org.myvoice.audio.AudioClip;
import org.myvoice.audio.Wav;
import org.myvoice.core.MiniJson;
import org.myvoice.lang.Languages;
import org.myvoice.mt.MarianTranslator;
import org.myvoice.mt.Prompts;
import org.myvoice.mt.UnigramTokenizer;
import org.myvoice.vad.SileroVad;

/**
 * The app's Java engines on this laptop with the real models, against reference values from the
 * Python tools (tools/make_test_vectors.py).
 */
public class ModelChecks {
    static Map<?, ?> golden() throws Exception {
        String g = System.getProperty("golden", "");
        assumeTrue("set MYVOICE_GOLDEN to the output of make_test_vectors.py", !g.isEmpty() && new File(g).isFile());
        try (FileReader r = new FileReader(g)) {
            return (Map<?, ?>) new MiniJson(r).parse();
        }
    }

    static File opus(String direction) {
        String d = System.getProperty("opus.dir", "");
        assumeTrue("set MYVOICE_OPUS_DIR", !d.isEmpty() && new File(d, direction + "/tokenizer.json").isFile());
        return new File(d, direction);
    }

    @Test
    public void tokenizerMatchesHuggingFace() throws Exception {
        Map<?, ?> tokens = (Map<?, ?>) golden().get("tokens");
        for (String direction : new String[] {"it-en", "en-it"}) {
            UnigramTokenizer tok;
            try (FileInputStream in = new FileInputStream(new File(opus(direction), "tokenizer.json"))) {
                tok = UnigramTokenizer.fromJson(in);
            }
            for (Map.Entry<?, ?> e : tokens.entrySet()) {
                String key = (String) e.getKey();
                if (!key.startsWith(direction + "|")) {
                    continue;
                }
                List<Integer> want = new ArrayList<>();
                for (Object o : (List<?>) e.getValue()) {
                    want.add(((Number) o).intValue());
                }
                assertEquals(key, want, tok.encode(key.substring(direction.length() + 1)));
            }
        }
    }

    @Test
    public void opusMtTranslatesLikePython() throws Exception {
        Map<?, ?> translations = (Map<?, ?>) golden().get("translations");
        try (MarianTranslator mt = new MarianTranslator(Map.of("it-en", opus("it-en"), "en-it", opus("en-it")), Backend.CPU, 4)) {
            for (Map.Entry<?, ?> e : translations.entrySet()) {
                String key = (String) e.getKey();
                String[] dir = key.substring(0, key.indexOf('|')).split("-");
                String text = key.substring(key.indexOf('|') + 1);
                long t0 = System.nanoTime();
                String out = mt.translate(text, dir[0], dir[1]);
                System.out.printf("%s -> %s (%.2f s)%n", text, out, (System.nanoTime() - t0) / 1e9);
                assertEquals(key, e.getValue(), out);
            }
        }
    }

    @Test
    public void sileroMatchesTheOfficialWrapper() throws Exception {
        List<?> want = (List<?>) golden().get("silero");
        AudioClip clip = Wav.read(new File(System.getProperty("clip.it"))).resampled(16000);
        try (SileroVad vad = new SileroVad(Files.readAllBytes(new File(System.getProperty("silero")).toPath()))) {
            float[] s = clip.samples();
            double maxDiff = 0;
            int n = 0;
            for (int i = 0; i + 512 <= s.length; i += 512, n++) {
                float[] f = new float[512];
                System.arraycopy(s, i, f, 0, 512);
                maxDiff = Math.max(maxDiff, Math.abs(vad.prob(f) - ((Number) want.get(n)).doubleValue()));
            }
            System.out.printf("silero: %d frames, max difference %.5f%n", n, maxDiff);
            assertEquals(want.size(), n);
            assertTrue("max difference " + maxDiff, maxDiff < 1e-3);
        }
    }

    @Test
    public void piperIdsMatchPiper() throws Exception {
        for (Object o : (List<?>) golden().get("piper")) {
            Map<?, ?> c = (Map<?, ?>) o;
            Map<?, ?> cfg;
            try (FileReader r = new FileReader((String) c.get("config"))) {
                cfg = (Map<?, ?>) new MiniJson(r).parse();
            }
            Map<String, long[]> idMap = org.myvoice.tts.PiperTts.readIdMap(cfg);
            String[] clauses = ((List<?>) c.get("clauses")).toArray(new String[0]);
            List<List<String>> sentences = org.myvoice.tts.EspeakPhonemizer.sentences(clauses);
            List<?> want = (List<?>) c.get("ids");
            assertEquals(c.get("text").toString(), want.size(), sentences.size());
            for (int i = 0; i < sentences.size(); i++) {
                List<?> w = (List<?>) want.get(i);
                long[] got = org.myvoice.tts.PiperTts.phonemeIds(idMap, sentences.get(i));
                long[] exp = new long[w.size()];
                for (int k = 0; k < exp.length; k++) {
                    exp[k] = ((Number) w.get(k)).longValue();
                }
                org.junit.Assert.assertArrayEquals(c.get("text") + " sentence " + i, exp, got);
            }
        }
    }

    @Test
    public void translateGemmaPromptMatchesTheModelsOwnTemplate() throws Exception {
        Map<?, ?> g = (Map<?, ?>) golden().get("gemma_prompt");
        String java = Prompts.translateGemma((String) g.get("text"), Languages.get("it"), Languages.get("en"));
        assertEquals(g.get("prompt"), java);
    }
}
