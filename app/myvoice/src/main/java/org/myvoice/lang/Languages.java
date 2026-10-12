package org.myvoice.lang;

import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Language table, translation-direction logic and a tiny text language guesser (port of langs.py). */
public final class Languages {
    private static final Map<String, Language> TABLE = new LinkedHashMap<>();

    static {
        for (Language l : new Language[] {
                new Language("it", "Italian", "it-IT"),
                new Language("en", "English", "en-US"),
                new Language("es", "Spanish", "es-ES"),
                new Language("fr", "French", "fr-FR"),
                new Language("de", "German", "de-DE"),
                new Language("pt", "Portuguese", "pt-PT"),
        }) {
            TABLE.put(l.code(), l);
        }
    }

    private Languages() {
    }

    public static Language get(String code) {
        Language l = TABLE.get(code);
        if (l == null) {
            throw new IllegalArgumentException("unsupported language '" + code + "' (known: " + TABLE.keySet() + ")");
        }
        return l;
    }

    public static String name(String code) {
        return get(code).name();
    }

    private static final Pattern DIRECTION = Pattern.compile("^\\s*([a-z]{2})\\s*(?:-|->|2|→|_to_)\\s*([a-z]{2})\\s*$");

    /** {src, tgt} for a fixed direction, or null for automatic detection ("auto"). */
    public static String[] parseDirection(String direction, List<String> languages) {
        if (direction.strip().equalsIgnoreCase("auto")) {
            return null;
        }
        Matcher m = DIRECTION.matcher(direction.toLowerCase(Locale.ROOT));
        if (!m.matches()) {
            throw new IllegalArgumentException("direction must be 'auto' or like 'it-en', got '" + direction + "'");
        }
        String src = m.group(1), tgt = m.group(2);
        if (!Set.of(src, tgt).equals(new HashSet<>(languages)) || src.equals(tgt)) {
            throw new IllegalArgumentException("direction '" + direction + "' does not match the language pair " + languages);
        }
        return new String[] {src, tgt};
    }

    public static String other(String src, List<String> languages) {
        if (src.equals(languages.get(0))) {
            return languages.get(1);
        }
        if (src.equals(languages.get(1))) {
            return languages.get(0);
        }
        throw new IllegalArgumentException("'" + src + "' is not in the language pair " + languages);
    }

    // Very small stop-word lists: enough to tell the languages of the pair apart in typed text.
    private static final Map<String, Set<String>> STOPWORDS = new HashMap<>();

    static {
        STOPWORDS.put("it", words("il lo la i gli le un una uno di da in con su per tra fra che non è e ed sono sei siamo "
                + "ho hai ha abbiamo avete hanno mi ti si ci vi del della dei delle al alla nel nella "
                + "questo questa quello quella come cosa perché quando dove anche più molto grazie ciao "
                + "buongiorno buonasera sì io tu lui lei noi voi loro mio mia tuo tua vorrei posso"));
        STOPWORDS.put("en", words("the a an of to in on at for with from by and or but not is are was were be been "
                + "have has had do does did i you he she we they it this that these those what which "
                + "who how why when where my your our their can could would should will please thanks "
                + "thank hello hi yes no"));
        STOPWORDS.put("es", words("el la los las un una de del que y en es son por para con no se lo le muy gracias hola"));
        STOPWORDS.put("fr", words("le la les un une des de du que et en est sont pour avec pas ne je tu il nous vous merci bonjour"));
        STOPWORDS.put("de", words("der die das ein eine und ist sind nicht ich du er sie wir ihr mit für von zu danke hallo"));
        STOPWORDS.put("pt", words("o a os as um uma de do da que e em é são para com não se eu você obrigado olá"));
    }

    private static Set<String> words(String s) {
        return new HashSet<>(List.of(s.split(" ")));
    }

    private static final Pattern WORD = Pattern.compile("[\\p{L}]+(?:'[\\p{L}]+)?");
    private static final Pattern IT_ACCENT = Pattern.compile("[àèéìòù]");

    /** Guess which of {@code candidates} a typed sentence is written in (stop-word vote). */
    public static String guessTextLanguage(String text, List<String> candidates) {
        Map<String, Integer> scores = new HashMap<>();
        for (String c : candidates) {
            scores.put(c, 0);
        }
        Matcher m = WORD.matcher(text);
        while (m.find()) {
            // Split Italian elisions: "l'acqua" -> "l", "acqua"
            for (String t : m.group().toLowerCase(Locale.ROOT).split("'")) {
                if (t.isEmpty()) {
                    continue;
                }
                for (String c : candidates) {
                    if (STOPWORDS.getOrDefault(c, Set.of()).contains(t)) {
                        scores.merge(c, 1, Integer::sum);
                    }
                }
            }
        }
        if (scores.containsKey("it")) {
            Matcher a = IT_ACCENT.matcher(text.toLowerCase(Locale.ROOT));
            int n = 0;
            while (a.find()) {
                n++;
            }
            scores.merge("it", 2 * n, Integer::sum);
        }
        String best = candidates.get(0);
        for (String c : candidates) {   // ties go to the earlier candidate
            if (scores.get(c) > scores.get(best)) {
                best = c;
            }
        }
        return best;
    }
}
