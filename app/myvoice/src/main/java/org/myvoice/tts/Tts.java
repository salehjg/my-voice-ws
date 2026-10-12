package org.myvoice.tts;

import org.myvoice.audio.AudioClip;

/** Text-to-speech: one voice per language. */
public interface Tts extends AutoCloseable {
    /** Speech for {@code text} in {@code lang}; empty audio for text with nothing to say. */
    AudioClip synthesize(String text, String lang) throws Exception;

    /** What it runs on, e.g. "System voices (Google)" or "Piper · CPU". */
    String describe();

    @Override
    default void close() {
    }
}
