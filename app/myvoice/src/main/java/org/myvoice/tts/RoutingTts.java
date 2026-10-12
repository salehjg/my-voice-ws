package org.myvoice.tts;

import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

import org.myvoice.audio.AudioClip;

/** Speaks each language with its own engine: the phone's voices or a Piper voice. */
public final class RoutingTts implements Tts {
    private final Map<String, Tts> byLang;

    public RoutingTts(Map<String, Tts> byLang) {
        this.byLang = byLang;
    }

    @Override
    public AudioClip synthesize(String text, String lang) throws Exception {
        Tts t = byLang.get(lang);
        if (t == null) {
            throw new IllegalStateException("no voice for " + lang);
        }
        return t.synthesize(text, lang);
    }

    @Override
    public String describe() {
        Set<String> parts = new LinkedHashSet<>();
        for (Tts t : byLang.values()) {
            parts.add(t.describe());
        }
        return String.join(" + ", parts);
    }

    @Override
    public void close() {
        for (Tts t : new LinkedHashSet<>(byLang.values())) {
            t.close();
        }
    }
}
