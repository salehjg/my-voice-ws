package org.myvoice.vad;

/** Speech probability for fixed-size frames of 16 kHz mono audio. */
public interface FrameVad extends AutoCloseable {
    int frameSize();

    float prob(float[] frame);

    void reset();

    @Override
    default void close() {
    }
}
