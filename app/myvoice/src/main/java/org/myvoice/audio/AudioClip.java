package org.myvoice.audio;

/** Mono float audio in [-1, 1] at a sample rate. */
public record AudioClip(float[] samples, int sampleRate) {
    public static AudioClip empty(int sampleRate) {
        return new AudioClip(new float[0], sampleRate);
    }

    public boolean isEmpty() {
        return samples.length == 0;
    }

    public double seconds() {
        return samples.length / (double) sampleRate;
    }

    /** Linear-interpolation resampling (good enough for speech going to the recogniser or a speaker). */
    public AudioClip resampled(int rate) {
        if (rate == sampleRate || samples.length == 0) {
            return new AudioClip(samples, rate);
        }
        int n = (int) Math.round(samples.length * (double) rate / sampleRate);
        float[] out = new float[n];
        double step = (double) sampleRate / rate;
        for (int i = 0; i < n; i++) {
            double x = i * step;
            int a = (int) x;
            int b = Math.min(a + 1, samples.length - 1);
            double f = x - a;
            out[i] = (float) (samples[Math.min(a, samples.length - 1)] * (1 - f) + samples[b] * f);
        }
        return new AudioClip(out, rate);
    }

    /** Concatenate clips at the first clip's rate with {@code gapS} of silence in between. */
    public static AudioClip concat(java.util.List<AudioClip> parts, double gapS) {
        java.util.List<AudioClip> ps = new java.util.ArrayList<>();
        for (AudioClip p : parts) {
            if (!p.isEmpty()) {
                ps.add(p);
            }
        }
        if (ps.isEmpty()) {
            return empty(22050);
        }
        int rate = ps.get(0).sampleRate();
        int gap = (int) (gapS * rate);
        int total = 0;
        java.util.List<float[]> chunks = new java.util.ArrayList<>();
        for (AudioClip p : ps) {
            float[] s = p.resampled(rate).samples();
            chunks.add(s);
            total += s.length;
        }
        total += gap * (chunks.size() - 1);
        float[] out = new float[total];
        int pos = 0;
        for (int i = 0; i < chunks.size(); i++) {
            if (i > 0) {
                pos += gap;
            }
            System.arraycopy(chunks.get(i), 0, out, pos, chunks.get(i).length);
            pos += chunks.get(i).length;
        }
        return new AudioClip(out, rate);
    }
}
