package org.myvoice.core;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.myvoice.asr.Asr;
import org.myvoice.asr.Transcript;
import org.myvoice.lang.Languages;
import org.myvoice.mt.Translator;
import org.myvoice.tts.Tts;
import org.myvoice.vad.Segmenter;

/** Holds the loaded models and runs one turn (utterance or text) through them (port of pipeline.Engine). */
public final class Engine implements AutoCloseable {
    public final Asr asr;
    public final Translator mt;
    public final Tts tts;
    public final List<String> languages;
    private final String[] fixed;

    public Engine(Settings settings, Asr asr, Translator mt, Tts tts) {
        this.asr = asr;
        this.mt = mt;
        this.tts = tts;
        this.languages = settings.languages;
        this.fixed = Languages.parseDirection(settings.direction, languages);
    }

    private static double since(long t0) {
        return (System.nanoTime() - t0) / 1e9;
    }

    /** Speech -> text, in {@code lang} if given (no detection), else detected. */
    public Transcript transcribe(float[] audio, String lang) throws Exception {
        if (lang == null && fixed != null) {
            lang = fixed[0];
        }
        return asr.transcribe(audio, languages, lang);
    }

    /** One utterance (16 kHz) -> transcript, translation (+ speech). {@code src} fixes the language. */
    public TurnResult processAudio(float[] audio, boolean speak, String src) throws Exception {
        Map<String, Double> timings = new LinkedHashMap<>();
        long t0 = System.nanoTime();
        Transcript tr = transcribe(audio, src);
        timings.put("asr", since(t0));
        if (tr == null) {
            return null;
        }
        String s, t;
        if (src != null) {
            s = src;
            t = Languages.other(src, languages);
        } else if (fixed != null) {
            s = fixed[0];
            t = fixed[1];
        } else {
            s = tr.language();
            t = Languages.other(s, languages);
        }
        t0 = System.nanoTime();
        String translation = mt.translate(tr.text(), s, t);
        timings.put("mt", since(t0));
        TurnResult r = new TurnResult(s, t, tr.text(), translation, tr.languageProb(),
                audio.length / (double) Segmenter.SAMPLE_RATE, timings);
        return finish(r, speak);
    }

    /** Text -> translation (+ speech). An explicit {@code src} wins over the direction setting. */
    public TurnResult processText(String text, String src, boolean speak) throws Exception {
        String s, t;
        if (src == null && fixed != null) {
            s = fixed[0];
            t = fixed[1];
        } else {
            s = src != null ? src : Languages.guessTextLanguage(text, languages);
            t = Languages.other(s, languages);
        }
        long t0 = System.nanoTime();
        String translation = mt.translate(text, s, t);
        Map<String, Double> timings = new LinkedHashMap<>();
        timings.put("mt", since(t0));
        return finish(new TurnResult(s, t, text, translation, 1.0f, 0.0, timings), speak);
    }

    private TurnResult finish(TurnResult r, boolean speak) throws Exception {
        if (r.tgtText == null || r.tgtText.isEmpty()) {
            return r;
        }
        r.tgtText = r.tgtText.strip();
        if (speak && tts != null) {
            long t0 = System.nanoTime();
            r.audio = tts.synthesize(r.tgtText, r.tgtLang);
            r.timings.put("tts", since(t0));
        }
        return r;
    }

    @Override
    public void close() throws Exception {
        if (asr != null) {
            asr.close();
        }
        if (mt != null) {
            mt.close();
        }
        if (tts != null) {
            tts.close();
        }
    }
}
