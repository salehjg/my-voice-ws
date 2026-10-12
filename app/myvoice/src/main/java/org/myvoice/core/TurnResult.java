package org.myvoice.core;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.StringJoiner;

import org.myvoice.audio.AudioClip;

/** One translated sentence (port of pipeline.TurnResult; the JSON has the same fields). */
public final class TurnResult {
    public final String srcLang;
    public final String tgtLang;
    public final String srcText;
    public String tgtText;
    public final float langProb;
    public final double utteranceS;
    public Map<String, Double> timings;
    public AudioClip audio = AudioClip.empty(22050);
    /** "They said", "You said", "Typed", "Redone", or "" in auto mode. */
    public String who = "";

    public TurnResult(String srcLang, String tgtLang, String srcText, String tgtText, float langProb,
                      double utteranceS, Map<String, Double> timings) {
        this.srcLang = srcLang;
        this.tgtLang = tgtLang;
        this.srcText = srcText;
        this.tgtText = tgtText;
        this.langProb = langProb;
        this.utteranceS = utteranceS;
        this.timings = timings != null ? timings : new LinkedHashMap<>();
    }

    /** "asr 0.5s · mt 0.4s" */
    public String timingText() {
        StringJoiner j = new StringJoiner(" · ");
        for (Map.Entry<String, Double> e : timings.entrySet()) {
            j.add(String.format(Locale.ROOT, "%s %.1fs", e.getKey(), e.getValue()));
        }
        return j.toString();
    }

    public String toJson() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("src_lang", srcLang);
        m.put("tgt_lang", tgtLang);
        m.put("src_text", srcText);
        m.put("tgt_text", tgtText);
        m.put("lang_prob", (double) langProb);
        m.put("utterance_s", utteranceS);
        m.put("timings", timings);
        m.put("sample_rate", audio.sampleRate());
        m.put("time", LocalDateTime.now().withNano(0).format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        return Json.object(m);
    }
}
