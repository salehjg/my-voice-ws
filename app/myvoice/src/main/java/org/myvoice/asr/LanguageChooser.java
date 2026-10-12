package org.myvoice.asr;

import java.util.List;
import java.util.Map;

/**
 * Whisper scores ~100 languages and short Italian clips are sometimes labelled Spanish or
 * Portuguese. Take the most likely language of the pair instead (port of asr.choose_language).
 */
public final class LanguageChooser {
    public record Choice(String language, float confidence) {
    }

    private LanguageChooser() {
    }

    /**
     * @param probs probability per language code (may be null or partial)
     * @return the best of {@code allowed}, with its probability renormalised over {@code allowed}
     */
    public static Choice choose(Map<String, Float> probs, String detected, float detectedProb, List<String> allowed) {
        double total = 0;
        String best = allowed.get(0);
        double bestP = -1;
        for (String lang : allowed) {
            double p = probs != null && probs.containsKey(lang) ? probs.get(lang)
                    : (probs == null && lang.equals(detected) ? detectedProb : 0.0);
            total += p;
            if (p > bestP) {
                bestP = p;
                best = lang;
            }
        }
        float conf = total > 0 ? (float) (bestP / total) : 1.0f / allowed.size();
        return new Choice(best, conf);
    }
}
