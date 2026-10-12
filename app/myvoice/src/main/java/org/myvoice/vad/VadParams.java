package org.myvoice.vad;

/** Segmenter settings (config.toml [vad]); {@code minSilenceMs} is what ends an utterance. */
public record VadParams(float threshold, int minSpeechMs, int minSilenceMs, int preRollMs, double maxUtteranceS) {
    public static VadParams defaults() {
        return new VadParams(0.5f, 250, 700, 300, 25.0);
    }

    public VadParams withMinSilenceMs(int ms) {
        return new VadParams(threshold, minSpeechMs, ms, preRollMs, maxUtteranceS);
    }

    public VadParams withMaxUtteranceS(double s) {
        return new VadParams(threshold, minSpeechMs, minSilenceMs, preRollMs, s);
    }
}
