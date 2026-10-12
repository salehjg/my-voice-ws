package org.myvoice.core;

import java.util.List;

import org.myvoice.accel.Choice;
import org.myvoice.accel.Stage;
import org.myvoice.asr.Asr;
import org.myvoice.asr.Transcript;
import org.myvoice.audio.AudioClip;
import org.myvoice.mt.Translator;
import org.myvoice.tts.Tts;

/**
 * Engines that fall back to the next backend when theirs fails: the chosen one first, then the
 * other working ones, the CPU last. The switch is reported, so the status always shows what is
 * really used.
 */
public final class Resilient {
    public interface Factory<T> {
        T create(Choice c) throws Exception;
    }

    public interface OnSwitch {
        void switched(Stage stage, Choice from, Choice to, String reason);
    }

    interface Call<T, R> {
        R apply(T engine) throws Exception;
    }

    static final class Holder<T extends AutoCloseable> {
        private final Stage stage;
        private final List<Choice> chain;
        private final Factory<T> factory;
        private final OnSwitch onSwitch;
        private int idx;
        private T current;

        Holder(Stage stage, List<Choice> chain, Factory<T> factory, OnSwitch onSwitch) throws Exception {
            this.stage = stage;
            this.chain = chain;
            this.factory = factory;
            this.onSwitch = onSwitch;
            open();
        }

        private static String why(Exception e) {
            return e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
        }

        private void open() throws Exception {
            Exception last = null;
            while (idx < chain.size()) {
                try {
                    current = factory.create(chain.get(idx));
                    return;
                } catch (Exception e) {
                    last = e;
                    if (idx + 1 < chain.size()) {
                        onSwitch.switched(stage, chain.get(idx), chain.get(idx + 1), why(e));
                    }
                    idx++;
                }
            }
            throw last != null ? last : new IllegalStateException("no backend for " + stage.label);
        }

        synchronized <R> R call(Call<T, R> fn) throws Exception {
            try {
                return fn.apply(current);
            } catch (Exception e) {
                if (idx + 1 >= chain.size()) {
                    throw e;
                }
                try {
                    current.close();
                } catch (Exception ignored) {
                }
                onSwitch.switched(stage, chain.get(idx), chain.get(idx + 1), why(e));
                idx++;
                open();
                return fn.apply(current);
            }
        }

        synchronized Choice choice() {
            return chain.get(Math.min(idx, chain.size() - 1));
        }

        synchronized T current() {
            return current;
        }

        synchronized void close() {
            try {
                if (current != null) {
                    current.close();
                }
            } catch (Exception ignored) {
            }
        }
    }

    private Resilient() {
    }

    public static Asr asr(Stage stage, List<Choice> chain, Factory<Asr> f, OnSwitch cb) throws Exception {
        Holder<Asr> h = new Holder<>(stage, chain, f, cb);
        return new Asr() {
            @Override
            public Transcript transcribe(float[] audio, List<String> allowed, String forced) throws Exception {
                return h.call(a -> a.transcribe(audio, allowed, forced));
            }

            @Override
            public String describe() {
                return h.current().describe();
            }

            @Override
            public void close() {
                h.close();
            }
        };
    }

    public static Translator translator(Stage stage, List<Choice> chain, Factory<Translator> f, OnSwitch cb) throws Exception {
        Holder<Translator> h = new Holder<>(stage, chain, f, cb);
        return new Translator() {
            @Override
            public String translate(String text, String src, String tgt) throws Exception {
                return h.call(t -> t.translate(text, src, tgt));
            }

            @Override
            public String describe() {
                return h.current().describe();
            }

            @Override
            public void close() {
                h.close();
            }
        };
    }

    public static Tts tts(Stage stage, List<Choice> chain, Factory<Tts> f, OnSwitch cb) throws Exception {
        Holder<Tts> h = new Holder<>(stage, chain, f, cb);
        return new Tts() {
            @Override
            public AudioClip synthesize(String text, String lang) throws Exception {
                return h.call(t -> t.synthesize(text, lang));
            }

            @Override
            public String describe() {
                return h.current().describe();
            }

            @Override
            public void close() {
                h.close();
            }
        };
    }
}
