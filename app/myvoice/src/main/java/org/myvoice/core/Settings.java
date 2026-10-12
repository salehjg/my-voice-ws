package org.myvoice.core;

import java.util.List;

import org.myvoice.vad.VadParams;

/**
 * Everything the conversation logic reads, with the laptop's config.toml defaults. The Android
 * side loads and saves it through {@code SettingsStore} (SharedPreferences, the counterpart of
 * config.local.toml).
 */
public final class Settings {
    // [general]
    public List<String> languages = List.of("it", "en");
    public String direction = "auto";          // "auto" | "it-en" | "en-it"
    public boolean sessionLog = true;

    // [talk]
    public String startMode = "auto";          // "auto" | "manual"
    public String theirLanguage = "it";
    public String myLanguage = "en";
    public boolean speakToMe = true;           // also speak translations into my language
    public boolean confirmMySpeech = true;     // manual mode: check what was heard before translating
    public int theirEndSilenceMs = 1000;
    public int myEndSilenceMs = 1500;
    public int autoEndSilenceMs = 1000;
    public double noSpeechTimeoutS = 15.0;
    public double maxRecordS = 60.0;

    // [audio]
    public boolean halfDuplex = true;          // pause the mic while speaking (false with headphones)
    public int playbackTailMs = 300;

    // [vad]
    public VadParams vad = VadParams.defaults();

    // [asr]
    public String initialPrompt = "";

    // [tts]
    public int sentenceSilenceMs = 150;

    public Settings copy() {
        Settings s = new Settings();
        s.languages = languages;
        s.direction = direction;
        s.sessionLog = sessionLog;
        s.startMode = startMode;
        s.theirLanguage = theirLanguage;
        s.myLanguage = myLanguage;
        s.speakToMe = speakToMe;
        s.confirmMySpeech = confirmMySpeech;
        s.theirEndSilenceMs = theirEndSilenceMs;
        s.myEndSilenceMs = myEndSilenceMs;
        s.autoEndSilenceMs = autoEndSilenceMs;
        s.noSpeechTimeoutS = noSpeechTimeoutS;
        s.maxRecordS = maxRecordS;
        s.halfDuplex = halfDuplex;
        s.playbackTailMs = playbackTailMs;
        s.vad = vad;
        s.initialPrompt = initialPrompt;
        s.sentenceSilenceMs = sentenceSilenceMs;
        return s;
    }
}
