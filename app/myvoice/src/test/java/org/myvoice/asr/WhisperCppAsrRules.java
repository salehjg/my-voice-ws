package org.myvoice.asr;

/** Test access to WhisperCppAsr's package-private rules. */
public final class WhisperCppAsrRules {
    private WhisperCppAsrRules() {
    }

    public static int audioCtxFor(int samples) {
        return WhisperCppAsr.audioCtxFor(samples);
    }

    public static String dropRepeatedSentences(String text) {
        return WhisperCppAsr.dropRepeatedSentences(text);
    }

    public static boolean looksRepeated(String text, int samples) {
        return WhisperCppAsr.looksRepeated(text, samples);
    }
}
