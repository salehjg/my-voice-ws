package org.myvoice.core;

import org.myvoice.audio.AudioClip;

/** Something the user asked for (a button, a notification action); the phone's version of the laptop's keys. */
public record Command(Kind kind, String text, String lang, AudioClip clip) {
    public enum Kind {
        // both modes
        EXIT,
        SET_MODE,          // text = "auto" | "manual"
        REPLAY,
        TOGGLE_SPEAK_TO_ME,
        TYPE,              // text, lang (null = guess the language)
        STOP_SPEAKING,
        // auto mode
        TOGGLE_PAUSE,
        PAUSE,
        RESUME,
        WRONG_LANGUAGE,
        // manual mode
        NEXT,              // SPACE: listen to them, or record my answer
        LISTEN_THEM,
        MY_TURN,
        SKIP,
        TOGGLE_CONFIRM,
        TRANSLATE_FILE,    // clip = 16 kHz audio, text = file name
        // while recording / checking my speech
        DONE,
        CANCEL,
        CONFIRM_OK,
        CONFIRM_EDIT,      // text = corrected sentence
        AGAIN
    }

    public static Command of(Kind kind) {
        return new Command(kind, null, null, null);
    }

    public static Command mode(String mode) {
        return new Command(Kind.SET_MODE, mode, null, null);
    }

    public static Command type(String text, String lang) {
        return new Command(Kind.TYPE, text, lang, null);
    }

    public static Command edit(String text) {
        return new Command(Kind.CONFIRM_EDIT, text, null, null);
    }

    public static Command file(String name, AudioClip clip16k) {
        return new Command(Kind.TRANSLATE_FILE, name, null, clip16k);
    }
}
