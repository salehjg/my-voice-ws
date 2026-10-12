package org.myvoice.audio;

/** A microphone: blocks of 16 kHz mono float audio, read from a queue while running. */
public interface MicSource extends AutoCloseable {
    void start() throws Exception;

    void stop();

    /** The next block, or null if none arrives within {@code timeoutMs}. */
    float[] read(long timeoutMs) throws InterruptedException;

    /** Throw away blocks that are already queued (stale audio from before a pause). */
    void drain();

    @Override
    default void close() {
        stop();
    }
}
