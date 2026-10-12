package org.myvoice;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import org.myvoice.asr.Asr;
import org.myvoice.asr.Transcript;
import org.myvoice.audio.AudioClip;
import org.myvoice.audio.AudioOut;
import org.myvoice.audio.MicSource;
import org.myvoice.core.Command;
import org.myvoice.core.Commands;
import org.myvoice.core.ConversationListener;
import org.myvoice.core.TurnResult;
import org.myvoice.mt.Translator;
import org.myvoice.tts.Tts;
import org.myvoice.vad.Segmenter;

/** Fake models, microphone, speaker and command sources (ports of the fakes in tests/test_talk.py). */
final class Fakes {
    private Fakes() {
    }

    static float[] tone(double seconds, float amp) {
        int n = (int) (seconds * Segmenter.SAMPLE_RATE);
        float[] t = new float[n];
        for (int i = 0; i < n; i++) {
            t[i] = (float) (amp * Math.sin(2 * Math.PI * 220 * i / Segmenter.SAMPLE_RATE));
        }
        return t;
    }

    static float[] tone(double seconds) {
        return tone(seconds, 0.3f);
    }

    static float[] silence(double seconds) {
        return new float[(int) (seconds * Segmenter.SAMPLE_RATE)];
    }

    static float[] concat(float[]... parts) {
        int n = 0;
        for (float[] p : parts) {
            n += p.length;
        }
        float[] out = new float[n];
        int pos = 0;
        for (float[] p : parts) {
            System.arraycopy(p, 0, out, pos, p.length);
            pos += p.length;
        }
        return out;
    }

    /** Each start() queues the next clip followed by 3 s of silence (cycling, or once each). */
    static class Mic implements MicSource {
        final List<float[]> clips;
        final boolean cycle;
        final LinkedBlockingQueue<float[]> blocks = new LinkedBlockingQueue<>();
        int n, starts, stops;

        Mic(List<float[]> clips, boolean cycle) {
            this.clips = clips;
            this.cycle = cycle;
        }

        float[] nextClip() {
            if (cycle || n < clips.size()) {
                return clips.get(n % clips.size());
            }
            return new float[0];
        }

        @Override
        public void start() {
            float[] clip = nextClip();
            n++;
            starts++;
            float[] audio = concat(new float[4000], clip, silence(3));
            for (int i = 0; i < audio.length; i += 512) {
                float[] b = new float[Math.min(512, audio.length - i)];
                System.arraycopy(audio, i, b, 0, b.length);
                blocks.add(b);
            }
        }

        @Override
        public void stop() {
            stops++;
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

    /** Speech too quiet for the energy VAD: nothing ends the recording except a command. */
    static class QuietMic extends Mic {
        QuietMic() {
            super(List.of(new float[0]), true);
        }

        @Override
        public void start() {
            starts++;
            for (int i = 0; i < 60; i++) {
                float[] b = new float[512];
                java.util.Arrays.fill(b, 0.001f);
                blocks.add(b);
            }
        }
    }

    static class FakeAsr implements Asr {
        final Map<String, String> texts = new HashMap<>(Map.of("it", "Buongiorno, come sta?", "en", "Very well, thank you."));
        final List<String> calls = new ArrayList<>();   // forced language per call ("null" = detect)
        final LinkedList<String> detect;
        final float prob;

        FakeAsr(List<String> detect, float prob) {
            this.detect = new LinkedList<>(detect);
            this.prob = prob;
        }

        FakeAsr() {
            this(List.of(), 0.99f);
        }

        @Override
        public Transcript transcribe(float[] audio, List<String> allowed, String forced) {
            calls.add(String.valueOf(forced));
            String lang = forced != null ? forced : (detect.isEmpty() ? "it" : detect.removeFirst());
            return new Transcript(texts.get(lang), lang, forced != null ? 1.0f : prob);
        }

        @Override
        public String describe() {
            return "fake ASR";
        }
    }

    static class FakeMt implements Translator {
        final List<String[]> calls = new ArrayList<>();

        @Override
        public String translate(String text, String src, String tgt) {
            calls.add(new String[] {text, src, tgt});
            return "<" + tgt + "> " + text;
        }

        @Override
        public String describe() {
            return "fake MT";
        }
    }

    static class FakeTts implements Tts {
        final List<String[]> spoken = new ArrayList<>();
        RuntimeException fail;

        @Override
        public AudioClip synthesize(String text, String lang) {
            if (fail != null) {
                throw fail;
            }
            spoken.add(new String[] {lang, text});
            float[] s = new float[2205];
            java.util.Arrays.fill(s, 0.1f);
            return new AudioClip(s, 22050);
        }

        @Override
        public String describe() {
            return "fake TTS";
        }
    }

    static class Player implements AudioOut {
        final List<Integer> played = new ArrayList<>();

        @Override
        public void start(AudioClip clip) {
            played.add(clip.samples().length);
        }

        @Override
        public boolean isPlaying() {
            return false;
        }

        @Override
        public void stop() {
        }
    }

    /** Manual-mode taps: polls (timeout >= 0) never take a command, so recordings end by silence. */
    static class Scripted implements Commands {
        final LinkedList<Command> queue;

        Scripted(List<Command> cmds) {
            queue = new LinkedList<>(cmds);
        }

        Command take() {
            return queue.isEmpty() ? Command.of(Command.Kind.EXIT) : queue.removeFirst();
        }

        @Override
        public Command next(long timeoutMs) {
            return timeoutMs < 0 ? take() : null;
        }

        @Override
        public void pushFront(Command c) {
            queue.addFirst(c);
        }

        @Override
        public void clear() {
        }
    }

    /** Auto mode only polls: the next command comes once the microphone has nothing left to hear. */
    static class WhenMicIdle extends Scripted {
        final Mic mic;

        WhenMicIdle(List<Command> cmds, Mic mic) {
            super(cmds);
            this.mic = mic;
        }

        @Override
        public Command next(long timeoutMs) {
            if (timeoutMs >= 0 && !mic.blocks.isEmpty()) {
                return null;
            }
            return take();
        }
    }

    /** Hands out {@code cmd} on the n-th poll (to stop a recording by hand), then the scripted ones. */
    static class OnPoll extends Scripted {
        int n;
        final Command cmd;

        OnPoll(int n, Command cmd, List<Command> rest) {
            super(rest);
            this.n = n;
            this.cmd = cmd;
        }

        @Override
        public Command next(long timeoutMs) {
            if (timeoutMs >= 0) {
                return --n == 0 ? cmd : null;
            }
            return take();
        }
    }

    /** Records everything the conversation reports. */
    static class Recorder implements ConversationListener {
        final List<String> modes = new ArrayList<>();
        final List<State> states = new ArrayList<>();
        final List<TurnResult> turns = new ArrayList<>();
        final List<String> notes = new ArrayList<>();
        final List<String> confirms = new ArrayList<>();
        final Map<String, Boolean> settings = new HashMap<>();

        @Override public void onMode(String mode) { modes.add(mode); }
        @Override public void onState(State state, String detail) { states.add(state); }
        @Override public void onLevel(double db, boolean speaking, double seconds) { }
        @Override public void onTurn(TurnResult r) { turns.add(r); }
        @Override public void onNote(Note kind, String text) { notes.add(text); }
        @Override public void onConfirm(String heard) { confirms.add(heard); }
        @Override public void onSettingChanged(String key, boolean value) { settings.put(key, value); }

        String allNotes() {
            return String.join("\n", notes);
        }
    }
}
