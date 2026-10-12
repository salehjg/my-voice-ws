package org.myvoice.audio;

import java.io.File;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Plays WAV files into the app as if they were spoken (debug setting "test audio input", like the
 * laptop's {@code --input-device file:a.wav,b.wav}): each start() uses the next file, in real time,
 * followed by silence.
 */
public final class FileMic implements MicSource {
    private static final int BLOCK = 512;
    private final List<File> files;
    private final LinkedBlockingQueue<float[]> blocks = new LinkedBlockingQueue<>();
    private int next;
    private Thread thread;
    private volatile boolean running;

    public FileMic(List<File> files) {
        if (files.isEmpty()) {
            throw new IllegalArgumentException("no test audio files");
        }
        this.files = files;
    }

    @Override
    public synchronized void start() throws Exception {
        AudioClip clip = Wav.read(files.get(next++ % files.size())).resampled(16000);
        running = true;
        thread = new Thread(() -> {
            float[] s = clip.samples();
            int pos = 0;
            while (running) {
                float[] b = new float[BLOCK];
                if (pos < s.length) {
                    System.arraycopy(s, pos, b, 0, Math.min(BLOCK, s.length - pos));
                }
                pos += BLOCK;
                blocks.offer(b);
                try {
                    Thread.sleep(BLOCK * 1000L / 16000);
                } catch (InterruptedException e) {
                    return;
                }
            }
        }, "file-mic");
        thread.start();
    }

    @Override
    public synchronized void stop() {
        running = false;
        if (thread != null) {
            thread.interrupt();
            try {
                thread.join(500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            thread = null;
        }
        blocks.clear();
    }

    @Override
    public float[] read(long timeoutMs) throws InterruptedException {
        return blocks.poll(timeoutMs, TimeUnit.MILLISECONDS);
    }

    @Override
    public void drain() {
        blocks.clear();
    }
}
