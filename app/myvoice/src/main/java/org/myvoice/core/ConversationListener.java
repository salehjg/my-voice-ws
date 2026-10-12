package org.myvoice.core;

/** What the conversation tells the UI and the notification. Called on the conversation thread. */
public interface ConversationListener {
    enum State {
        STARTING,
        LISTENING,      // auto: waiting for speech
        HEARING,        // auto: someone is talking
        PAUSED,         // auto: mic off
        TRANSLATING,
        SPEAKING,
        READY,          // manual: "listen to them" is next
        RECORDING,      // manual: detail = language name
        CONFIRM,        // manual: check what was heard from me
        REPLY,          // manual: "your turn"
        STOPPED
    }

    enum Note { INFO, WARNING, ERROR }

    void onMode(String mode);

    void onState(State state, String detail);

    /** Microphone level for the meter (about 15 times a second while listening). */
    void onLevel(double db, boolean speaking, double seconds);

    void onTurn(TurnResult r);

    void onNote(Note kind, String text);

    /** Manual mode check step: show {@code heard}, then send CONFIRM_OK / CONFIRM_EDIT / AGAIN / CANCEL. */
    void onConfirm(String heard);

    /** A setting changed from inside the conversation (A / C toggles); the app saves it. */
    void onSettingChanged(String key, boolean value);

    /** Silent listener, for tests and the probe. */
    ConversationListener NONE = new ConversationListener() {
        @Override public void onMode(String mode) { }
        @Override public void onState(State state, String detail) { }
        @Override public void onLevel(double db, boolean speaking, double seconds) { }
        @Override public void onTurn(TurnResult r) { }
        @Override public void onNote(Note kind, String text) { }
        @Override public void onConfirm(String heard) { }
        @Override public void onSettingChanged(String key, boolean value) { }
    };
}
