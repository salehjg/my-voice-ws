package org.myvoice.asr;

import java.util.List;

/** Speech recogniser: 16 kHz mono float audio -> text in one of {@code allowed}. */
public interface Asr extends AutoCloseable {
    /**
     * @param forcedLanguage language to decode in, or null to detect it (restricted to {@code allowed})
     * @return null if nothing usable was said
     */
    Transcript transcribe(float[] audio, List<String> allowed, String forcedLanguage) throws Exception;

    /** What it runs on, e.g. "Whisper small · CPU ×4". */
    String describe();

    @Override
    default void close() {
    }
}
