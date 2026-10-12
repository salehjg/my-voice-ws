package org.myvoice.tts;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import android.content.Context;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.speech.tts.Voice;

import org.myvoice.audio.AudioClip;
import org.myvoice.audio.Wav;

/**
 * Android's own offline voices (usually Google's speech engine). Each sentence is rendered to a
 * WAV file and played by the app, so replay, "any tap stops it" and pausing the mic work the same
 * as with Piper.
 */
public final class SystemTts implements Tts {
    public static final String PREFIX = "system:";
    private static final long TIMEOUT_S = 30;

    /** Google voice names whose gender is known (the engine doesn't say); used to default to a male voice. */
    private static final Map<String, String> GENDER = Map.ofEntries(
            Map.entry("en-us-x-iol-local", "male"), Map.entry("en-us-x-iom-local", "male"),
            Map.entry("en-us-x-tpd-local", "male"), Map.entry("en-gb-x-gbd-local", "male"),
            Map.entry("en-gb-x-rjs-local", "male"), Map.entry("en-us-x-iob-local", "female"),
            Map.entry("en-us-x-iog-local", "female"), Map.entry("en-us-x-sfg-local", "female"),
            Map.entry("en-us-x-tpc-local", "female"), Map.entry("en-us-x-tpf-local", "female"),
            Map.entry("en-gb-x-gba-local", "female"), Map.entry("en-gb-x-gbc-local", "female"),
            Map.entry("it-it-x-itc-local", "male"), Map.entry("it-it-x-itd-local", "male"),
            Map.entry("it-it-x-itb-local", "female"), Map.entry("it-it-x-kda-local", "female"));
    private static final Map<String, List<String>> PREFERRED_MALE = Map.of(
            "en", List.of("en-us-x-iom-local", "en-us-x-iol-local", "en-us-x-tpd-local", "en-gb-x-rjs-local"),
            "it", List.of("it-it-x-itd-local", "it-it-x-itc-local"));

    private final TextToSpeech tts;
    private final File cacheDir;
    private final Map<String, String> chosen = new HashMap<>();   // language -> voice name ("" = default)
    private final Map<String, CountDownLatch> pending = Collections.synchronizedMap(new HashMap<>());
    private final Map<String, Boolean> failed = Collections.synchronizedMap(new HashMap<>());
    private int counter;

    /** Blocks until the engine is ready (up to 10 s). */
    public SystemTts(Context context, Map<String, String> voiceByLang) throws Exception {
        CountDownLatch ready = new CountDownLatch(1);
        int[] status = {TextToSpeech.ERROR};
        tts = new TextToSpeech(context.getApplicationContext(), s -> {
            status[0] = s;
            ready.countDown();
        });
        if (!ready.await(10, TimeUnit.SECONDS) || status[0] != TextToSpeech.SUCCESS) {
            tts.shutdown();
            throw new IllegalStateException("the phone's text-to-speech engine did not start");
        }
        cacheDir = new File(context.getCacheDir(), "tts");
        cacheDir.mkdirs();
        tts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
            @Override
            public void onStart(String id) {
            }

            @Override
            public void onDone(String id) {
                release(id, false);
            }

            @Override
            @SuppressWarnings("deprecation")
            public void onError(String id) {
                release(id, true);
            }

            @Override
            public void onError(String id, int code) {
                release(id, true);
            }
        });
        for (Map.Entry<String, String> e : voiceByLang.entrySet()) {
            chosen.put(e.getKey(), e.getValue() == null ? "" : e.getValue().replace(PREFIX, ""));
        }
    }

    private void release(String id, boolean error) {
        failed.put(id, error);
        CountDownLatch l = pending.get(id);
        if (l != null) {
            l.countDown();
        }
    }

    /** Offline voices for a language, best first (male voices first when known). */
    public List<Voice> voices(String lang) {
        List<Voice> out = new ArrayList<>();
        Set<Voice> all = tts.getVoices();
        if (all == null) {
            return out;
        }
        for (Voice v : all) {
            if (!v.getLocale().getLanguage().equals(lang) || v.isNetworkConnectionRequired()) {
                continue;
            }
            if (v.getFeatures() != null && v.getFeatures().contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED)) {
                continue;
            }
            out.add(v);
        }
        out.sort((a, b) -> Integer.compare(rank(lang, a.getName()), rank(lang, b.getName())));
        return out;
    }

    private static int rank(String lang, String name) {
        int i = PREFERRED_MALE.getOrDefault(lang, List.of()).indexOf(name);
        return i >= 0 ? i : 100 + ("male".equals(GENDER.get(name)) ? 0 : 1);
    }

    public static String gender(String voiceName) {
        return GENDER.getOrDefault(voiceName, "");
    }

    public String engineName() {
        String e = tts.getDefaultEngine();
        return e == null ? "system" : e;
    }

    private Voice voiceFor(String lang) {
        String want = chosen.getOrDefault(lang, "");
        List<Voice> vs = voices(lang);
        for (Voice v : vs) {
            if (v.getName().equals(want)) {
                return v;
            }
        }
        return vs.isEmpty() ? null : vs.get(0);
    }

    public void setVoice(String lang, String voiceId) {
        chosen.put(lang, voiceId.replace(PREFIX, ""));
    }

    @Override
    public synchronized AudioClip synthesize(String text, String lang) throws Exception {
        if (text.codePoints().noneMatch(Character::isLetterOrDigit)) {
            return AudioClip.empty(22050);
        }
        Voice v = voiceFor(lang);
        if (v != null) {
            tts.setVoice(v);
        } else if (tts.setLanguage(Locale.forLanguageTag(lang)) < TextToSpeech.LANG_AVAILABLE) {
            throw new IllegalStateException("no offline " + new Locale(lang).getDisplayLanguage(Locale.ENGLISH)
                    + " voice: install it in Android settings → Text-to-speech");
        }
        String id = "u" + (++counter);
        File out = new File(cacheDir, id + ".wav");
        CountDownLatch done = new CountDownLatch(1);
        pending.put(id, done);
        try {
            int rc = tts.synthesizeToFile(text, new Bundle(), out, id);
            if (rc != TextToSpeech.SUCCESS) {
                throw new IllegalStateException("text-to-speech failed (" + rc + ")");
            }
            if (!done.await(TIMEOUT_S, TimeUnit.SECONDS)) {
                throw new IllegalStateException("text-to-speech timed out");
            }
            if (Boolean.TRUE.equals(failed.get(id))) {
                throw new IllegalStateException("text-to-speech engine reported an error");
            }
            return Wav.read(out);
        } finally {
            pending.remove(id);
            failed.remove(id);
            out.delete();
        }
    }

    @Override
    public String describe() {
        return "System voices (" + engineName() + ")";
    }

    @Override
    public void close() {
        tts.shutdown();
    }
}
