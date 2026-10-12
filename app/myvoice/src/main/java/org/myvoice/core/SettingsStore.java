package org.myvoice.core;

import android.content.Context;
import android.content.SharedPreferences;

import androidx.preference.PreferenceManager;

import org.myvoice.vad.VadParams;

/**
 * Settings and choices saved on the phone (the counterpart of the laptop's config.local.toml).
 * Keys match res/xml/preferences.xml.
 */
public final class SettingsStore {
    public static final String START_MODE = "start_mode";
    public static final String SPEAK_TO_ME = "speak_to_me";
    public static final String CONFIRM_MY_SPEECH = "confirm_my_speech";
    public static final String AUTO_END_SILENCE = "auto_end_silence_ms";
    public static final String THEIR_END_SILENCE = "their_end_silence_ms";
    public static final String MY_END_SILENCE = "my_end_silence_ms";
    public static final String LISTENING = "listening";            // "auto" | "speakers" | "headphones"
    public static final String VAD_THRESHOLD = "vad_threshold";
    public static final String INITIAL_PROMPT = "initial_prompt";
    public static final String DIRECTION = "direction";
    public static final String SESSION_LOG = "session_log";
    public static final String SHORT_AUDIO_CTX = "short_audio_ctx";
    public static final String VOICE_PREFIX = "voice_";            // voice_it / voice_en: "system:<name>" | "piper:<model id>"
    public static final String MODEL_PREFIX = "model_";            // model_asr / model_mt / model_voice_it / model_voice_en
    public static final String ACCEL_PREFIX = "accel_";            // accel_asr / accel_mt / accel_tts: "auto" | "cpu" | backend
    public static final String TEST_AUDIO = "test_audio_input";
    public static final String WIFI_ONLY = "wifi_only";
    public static final String SETUP_DONE = "setup_done";

    private SettingsStore() {
    }

    public static SharedPreferences prefs(Context c) {
        return PreferenceManager.getDefaultSharedPreferences(c.getApplicationContext());
    }

    private static int intPref(SharedPreferences p, String key, int def) {
        try {
            return Integer.parseInt(p.getString(key, Integer.toString(def)).trim());
        } catch (NumberFormatException | ClassCastException e) {
            return def;
        }
    }

    /** The conversation settings; {@code headphonesConnected} decides "auto" listening. */
    public static Settings load(Context c, boolean headphonesConnected) {
        SharedPreferences p = prefs(c);
        Settings s = new Settings();
        s.startMode = p.getString(START_MODE, "auto");
        s.speakToMe = p.getBoolean(SPEAK_TO_ME, true);
        s.confirmMySpeech = p.getBoolean(CONFIRM_MY_SPEECH, true);
        s.autoEndSilenceMs = intPref(p, AUTO_END_SILENCE, 1000);
        s.theirEndSilenceMs = intPref(p, THEIR_END_SILENCE, 1000);
        s.myEndSilenceMs = intPref(p, MY_END_SILENCE, 1500);
        String listening = p.getString(LISTENING, "auto");
        s.halfDuplex = switch (listening) {
            case "headphones" -> false;
            case "speakers" -> true;
            default -> !headphonesConnected;
        };
        float threshold;
        try {
            threshold = Float.parseFloat(p.getString(VAD_THRESHOLD, "0.5"));
        } catch (NumberFormatException e) {
            threshold = 0.5f;
        }
        VadParams d = VadParams.defaults();
        s.vad = new VadParams(threshold, d.minSpeechMs(), d.minSilenceMs(), d.preRollMs(), d.maxUtteranceS());
        s.initialPrompt = p.getString(INITIAL_PROMPT, "");
        s.direction = p.getString(DIRECTION, "auto");
        s.sessionLog = p.getBoolean(SESSION_LOG, true);
        return s;
    }

    public static void setBoolean(Context c, String key, boolean value) {
        prefs(c).edit().putBoolean(key, value).apply();
    }

    public static String get(Context c, String key, String def) {
        return prefs(c).getString(key, def);
    }

    public static void put(Context c, String key, String value) {
        prefs(c).edit().putString(key, value).apply();
    }
}
