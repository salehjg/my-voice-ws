package org.myvoice.vad;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

/**
 * Turns a continuous 16 kHz stream into utterances (port of vad.py): it opens when the VAD
 * probability crosses the threshold, keeps {@code preRollMs} of audio from before the onset, and
 * closes after {@code minSilenceMs} of silence (or at {@code maxUtteranceS}).
 */
public final class Segmenter {
    public static final int SAMPLE_RATE = 16000;
    private static final int TAIL_KEEP_MS = 200;   // trailing silence kept on each utterance

    private final FrameVad vad;
    private final int frame;
    private final float threshold;
    private final float negThreshold;
    private final int preRollFrames;
    private final int minSpeechFrames;
    private final int minSilenceFrames;
    private final int tailKeepFrames;
    private final int maxFrames;

    private float[] pending = new float[0];
    private long framesSeen;
    private ArrayDeque<float[]> preroll;
    private List<float[]> frames;
    private boolean triggered;
    private int speech;
    private int silence;
    private long startFrame;

    public Segmenter(FrameVad vad, VadParams p) {
        this.vad = vad;
        this.frame = vad.frameSize();
        double frameMs = frame * 1000.0 / SAMPLE_RATE;
        this.threshold = p.threshold();
        this.negThreshold = Math.max(p.threshold() - 0.15f, 0.01f);
        this.preRollFrames = Math.max(0, (int) Math.ceil(p.preRollMs() / frameMs));
        this.minSpeechFrames = Math.max(1, (int) Math.ceil(p.minSpeechMs() / frameMs));
        this.minSilenceFrames = Math.max(1, (int) Math.ceil(p.minSilenceMs() / frameMs));
        this.tailKeepFrames = Math.min(minSilenceFrames, (int) Math.ceil(TAIL_KEEP_MS / frameMs));
        this.maxFrames = Math.max(minSpeechFrames + 1, (int) (p.maxUtteranceS() * 1000 / frameMs));
        clear();
    }

    private void clear() {
        preroll = new ArrayDeque<>(preRollFrames + 1);
        frames = new ArrayList<>();
        triggered = false;
        speech = 0;
        silence = 0;
        startFrame = 0;
    }

    public boolean inSpeech() {
        return triggered;
    }

    /** Drop everything buffered (used while the mic is paused during playback). */
    public void reset() {
        pending = new float[0];
        clear();
        vad.reset();
    }

    public List<Utterance> feed(float[] samples) {
        float[] buf = new float[pending.length + samples.length];
        System.arraycopy(pending, 0, buf, 0, pending.length);
        System.arraycopy(samples, 0, buf, pending.length, samples.length);
        int n = buf.length / frame;
        List<Utterance> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            float[] f = new float[frame];
            System.arraycopy(buf, i * frame, f, 0, frame);
            Utterance u = step(f);
            if (u != null) {
                out.add(u);
            }
        }
        int rest = buf.length - n * frame;
        pending = new float[rest];
        System.arraycopy(buf, n * frame, pending, 0, rest);
        return out;
    }

    /** End of stream: emit the utterance in progress, if it is long enough. */
    public Utterance flush() {
        Utterance u = triggered ? finish(false) : null;
        reset();
        return u;
    }

    private Utterance step(float[] f) {
        float p = vad.prob(f);
        framesSeen++;
        if (!triggered) {
            if (p >= threshold) {
                triggered = true;
                frames = new ArrayList<>(preroll);
                frames.add(f);
                startFrame = framesSeen - frames.size();
                speech = 1;
                silence = 0;
            } else {
                if (preRollFrames > 0) {
                    if (preroll.size() == preRollFrames) {
                        preroll.removeFirst();
                    }
                    preroll.addLast(f);
                }
            }
            return null;
        }
        frames.add(f);
        if (p >= threshold) {
            speech++;
            silence = 0;
        } else if (p < negThreshold || silence > 0) {
            // Below negThreshold starts the silence timer; the hysteresis band only extends it.
            silence++;
        }
        if (silence >= minSilenceFrames) {
            return finish(false);
        }
        if (frames.size() >= maxFrames) {
            return finish(true);
        }
        return null;
    }

    private Utterance finish(boolean forced) {
        List<float[]> fs = frames;
        if (!forced && silence > tailKeepFrames) {
            fs = fs.subList(0, fs.size() - (silence - tailKeepFrames));
        }
        boolean enough = speech >= minSpeechFrames;
        double start = startFrame * (double) frame / SAMPLE_RATE;
        Utterance u = null;
        if (enough && !fs.isEmpty()) {
            float[] audio = new float[fs.size() * frame];
            for (int i = 0; i < fs.size(); i++) {
                System.arraycopy(fs.get(i), 0, audio, i * frame, frame);
            }
            u = new Utterance(audio, start, start + audio.length / (double) SAMPLE_RATE, forced);
        }
        clear();
        vad.reset();
        return u;
    }

    /** Audio level of a block in dB (for the level meter). */
    public static double levelDb(float[] block) {
        if (block.length == 0) {
            return -180;
        }
        double sum = 0;
        for (float v : block) {
            sum += (double) v * v;
        }
        return 20.0 * Math.log10(Math.sqrt(sum / block.length) + 1e-9);
    }
}
