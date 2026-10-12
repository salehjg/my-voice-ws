package org.myvoice.vad;

/** One stretch of speech (float mono, 16 kHz). */
public record Utterance(float[] audio, double startS, double endS, boolean forcedCut) {
    public double durationS() {
        return audio.length / (double) Segmenter.SAMPLE_RATE;
    }
}
