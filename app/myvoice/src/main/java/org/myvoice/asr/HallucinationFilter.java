package org.myvoice.asr;

import java.text.Normalizer;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;

/** Phrases Whisper is known to "hear" in silence or noise (port of asr.is_hallucination). */
public final class HallucinationFilter {
    private static final List<String> SUBSTRINGS = List.of(
            "amara.org",
            "sottotitoli creati dalla comunità",
            "sottotitoli a cura di",
            "sottotitoli e revisione a cura di",
            "subtitles by the amara",
            "qtss");
    private static final Set<String> EXACT = Set.of(
            "thank you for watching",
            "thanks for watching",
            "thank you for watching and see you next time",
            "please subscribe",
            "subscribe to the channel",
            "grazie per la visione",
            "grazie a tutti per la visione",
            "iscriviti al canale",
            "you");
    private static final Pattern SPACES = Pattern.compile("\\s+");
    private static final Pattern WORD_CHAR = Pattern.compile("[\\p{L}\\p{N}_]");
    private static final String STRIP = " .!?…,;:-–—\"'«»“”";

    private HallucinationFilter() {
    }

    static String normalise(String text) {
        String t = Normalizer.normalize(text, Normalizer.Form.NFKC).toLowerCase(Locale.ROOT).strip();
        t = SPACES.matcher(t).replaceAll(" ");
        int a = 0, b = t.length();
        while (a < b && STRIP.indexOf(t.charAt(a)) >= 0) {
            a++;
        }
        while (b > a && STRIP.indexOf(t.charAt(b - 1)) >= 0) {
            b--;
        }
        return t.substring(a, b);
    }

    public static boolean isHallucination(String text) {
        String norm = normalise(text);
        if (norm.isEmpty() || !WORD_CHAR.matcher(norm).find()) {
            return true;
        }
        for (String s : SUBSTRINGS) {
            if (norm.contains(s)) {
                return true;
            }
        }
        return EXACT.contains(norm);
    }
}
