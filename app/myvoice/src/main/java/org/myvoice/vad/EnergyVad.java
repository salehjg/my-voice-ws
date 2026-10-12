package org.myvoice.vad;

/** RMS-energy VAD: a dependency-free fallback (fine in a quiet room, poor in noise). */
public final class EnergyVad implements FrameVad {
    private final double thresholdDb;

    public EnergyVad(double thresholdDb) {
        this.thresholdDb = thresholdDb;
    }

    @Override
    public int frameSize() {
        return 512;
    }

    @Override
    public float prob(float[] frame) {
        double sum = 0;
        for (float v : frame) {
            sum += (double) v * v;
        }
        double rms = Math.sqrt(sum / frame.length) + 1e-10;
        double db = 20.0 * Math.log10(rms);
        return (float) (1.0 / (1.0 + Math.exp(-(db - thresholdDb) / 3.0)));
    }

    @Override
    public void reset() {
    }
}
