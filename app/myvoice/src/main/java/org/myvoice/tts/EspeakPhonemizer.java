package org.myvoice.tts;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.List;

import org.myvoice.nativebridge.NativeLib;

/**
 * Text -> IPA phonemes with espeak-ng, grouped by sentence, exactly like Piper's own
 * {@code phonemize_espeak.py}: clause punctuation is kept, "(lang)" switch flags are removed,
 * and phonemes are split into NFD code points.
 */
public final class EspeakPhonemizer {
    private static boolean initialised;

    /** @param dataDir folder that contains espeak-ng-data (extracted from the APK assets) */
    public static synchronized void init(String dataDir) {
        if (!initialised) {
            int rc = NativeLib.espeakInit(dataDir);
            if (rc != 0) {
                throw new IllegalStateException("espeak-ng could not start (error " + rc + ")");
            }
            initialised = true;
        }
    }

    /** @param voice espeak voice name from the Piper voice config, e.g. "it" or "en-us" */
    public static synchronized List<List<String>> phonemize(String voice, String text) {
        if (NativeLib.espeakSetVoice(voice) != 0) {
            throw new IllegalArgumentException("espeak-ng has no voice '" + voice + "'");
        }
        // One string per clause: phonemes \u0001 terminator \u0001 end-of-sentence(0/1)
        return sentences(NativeLib.espeakPhonemize(text));
    }

    /** Piper's grouping of espeak's clauses into sentences of single phonemes (phonemize_espeak.py). */
    public static List<List<String>> sentences(String[] clauses) {
        List<List<String>> all = new ArrayList<>();
        List<String> sentence = new ArrayList<>();
        for (String c : clauses) {
            String[] f = c.split("\u0001", -1);
            String ph = f[0].replaceAll("\\([^)]+\\)", "");
            String term = f.length > 1 ? f[1] : "";
            boolean end = f.length > 2 && f[2].equals("1");
            ph += term;
            if (term.equals(",") || term.equals(":") || term.equals(";")) {
                ph += " ";   // not a sentence boundary
            }
            String nfd = Normalizer.normalize(ph, Normalizer.Form.NFD);
            nfd.codePoints().forEach(cp -> sentence.add(new String(Character.toChars(cp))));
            if (end) {
                all.add(new ArrayList<>(sentence));
                sentence.clear();
            }
        }
        if (!sentence.isEmpty()) {
            all.add(sentence);
        }
        return all;
    }

    private EspeakPhonemizer() {
    }
}
