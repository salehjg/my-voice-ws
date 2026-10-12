package org.myvoice.audio;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import android.annotation.SuppressLint;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.audiofx.AcousticEchoCanceler;

/**
 * The phone's microphone: 16 kHz mono, read on a background thread in 32 ms blocks.
 * {@code VOICE_RECOGNITION} keeps the signal clean for speech recognition; with
 * {@code echoCancel} it uses {@code VOICE_COMMUNICATION} plus the platform echo canceller instead.
 */
public final class AudioRecordMic implements MicSource {
    public static final int RATE = 16000;
    private static final int BLOCK = 512;
    private final boolean echoCancel;
    private final LinkedBlockingQueue<float[]> blocks = new LinkedBlockingQueue<>(2000);
    private AudioRecord rec;
    private AcousticEchoCanceler aec;
    private Thread reader;
    private volatile boolean running;
    public volatile int overflows;

    public AudioRecordMic(boolean echoCancel) {
        this.echoCancel = echoCancel;
    }

    @Override
    @SuppressLint("MissingPermission")   // RECORD_AUDIO is requested before the conversation starts
    public synchronized void start() {
        if (running) {
            return;
        }
        int min = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        rec = new AudioRecord.Builder()
                .setAudioSource(echoCancel ? MediaRecorder.AudioSource.VOICE_COMMUNICATION : MediaRecorder.AudioSource.VOICE_RECOGNITION)
                .setAudioFormat(new AudioFormat.Builder().setSampleRate(RATE)
                        .setChannelMask(AudioFormat.CHANNEL_IN_MONO).setEncoding(AudioFormat.ENCODING_PCM_16BIT).build())
                .setBufferSizeInBytes(Math.max(min, RATE * 2))
                .build();
        if (rec.getState() != AudioRecord.STATE_INITIALIZED) {
            rec.release();
            rec = null;
            throw new IllegalStateException("could not open the microphone (is another app using it?)");
        }
        if (echoCancel && AcousticEchoCanceler.isAvailable()) {
            aec = AcousticEchoCanceler.create(rec.getAudioSessionId());
            if (aec != null) {
                aec.setEnabled(true);
            }
        }
        rec.startRecording();
        running = true;
        final AudioRecord r = rec;
        reader = new Thread(() -> {
            short[] buf = new short[BLOCK];
            while (running) {
                int n = r.read(buf, 0, BLOCK, AudioRecord.READ_BLOCKING);
                if (n < 0) {
                    break;
                }
                if (n == 0) {
                    continue;
                }
                float[] f = new float[n];
                for (int i = 0; i < n; i++) {
                    f[i] = buf[i] / 32768f;
                }
                if (!blocks.offer(f)) {
                    overflows++;   // consumer stalled; drop audio rather than block the reader
                }
            }
        }, "mic");
        reader.start();
    }

    @Override
    public synchronized void stop() {
        if (!running && rec == null) {
            return;
        }
        running = false;
        if (rec != null) {
            try {
                rec.stop();
            } catch (IllegalStateException ignored) {
            }
        }
        if (reader != null) {
            try {
                reader.join(500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            reader = null;
        }
        if (aec != null) {
            aec.release();
            aec = null;
        }
        if (rec != null) {
            rec.release();
            rec = null;
        }
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
