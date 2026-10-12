package org.myvoice.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

import org.myvoice.asr.Transcript;
import org.myvoice.audio.AudioClip;
import org.myvoice.audio.AudioOut;
import org.myvoice.audio.MicSource;
import org.myvoice.core.Command.Kind;
import org.myvoice.core.ConversationListener.Note;
import org.myvoice.core.ConversationListener.State;
import org.myvoice.lang.Languages;
import org.myvoice.tts.Tts;
import org.myvoice.vad.FrameVad;
import org.myvoice.vad.Segmenter;
import org.myvoice.vad.Utterance;
import org.myvoice.vad.VadParams;

/**
 * The conversation, in two modes (port of the laptop's talk.py).
 *
 * <p>Auto mode: the microphone is always on; every sentence is recognised as Italian or English,
 * translated into the other language, shown and spoken. With speakers the mic is paused while a
 * translation is spoken, so the app doesn't translate itself.
 *
 * <p>Manual mode: they speak (Italian), you read/hear it in English, you answer in English and it is
 * spoken in Italian; buttons start and skip each step.
 *
 * <p>Runs on its own thread; {@link Commands} carry the user's taps in, {@link ConversationListener}
 * carries everything the UI shows out.
 */
public final class Conversation {
    public static final Pattern SENTENCE_END = Pattern.compile("(?<=[.!?…])\\s+(?=\\S)");
    private static final String AGAIN = "\u0000again";
    private static final long LEVEL_EVERY_MS = 66;

    private final Engine engine;
    private final Settings s;
    private final Commands commands;
    private final ConversationListener ui;
    private final AudioOut player;
    private final MicSource mic;
    private final FrameVad vad;
    private final SessionLog log;
    private final String their;
    private final String mine;

    private String mode = "manual";
    private AudioClip lastAudio;
    private float[] lastHeardAudio;   // auto mode: the last sentence, for "wrong language"
    private String lastHeardLang;
    private boolean micOn;
    private boolean restartMic;
    private long lastLevelMs;
    public final List<TurnResult> history = new ArrayList<>();

    public Conversation(Engine engine, Settings settings, Commands commands, ConversationListener ui,
                        AudioOut player, MicSource mic, FrameVad vad, SessionLog log) {
        this.engine = engine;
        this.s = settings;
        this.commands = commands;
        this.ui = ui;
        this.player = player;
        this.mic = mic;
        this.vad = vad;
        this.log = log;
        this.their = settings.theirLanguage;
        this.mine = settings.myLanguage;
    }

    public String mode() {
        return mode;
    }

    private static String name(String lang) {
        return Languages.name(lang);
    }

    // ------------------------------------------------------------------ main loop

    /** Run until EXIT; {@code startMode} is "auto" or "manual". */
    public void run(String startMode) throws InterruptedException {
        String next = startMode;
        try {
            while (!next.equals("quit")) {
                mode = next;
                ui.onMode(mode);
                next = mode.equals("auto") ? runAuto() : runManual();
            }
        } finally {
            stopMic();
            player.stop();
            ui.onState(State.STOPPED, "");
        }
    }

    private void error(Exception e) {
        String msg = e.getMessage() != null ? e.getMessage() : "";
        ui.onNote(Note.ERROR, "something went wrong: " + e.getClass().getSimpleName() + (msg.isEmpty() ? "" : ": " + msg));
    }

    // ------------------------------------------------------------------ audio out

    /** Auto mode with speakers: stop listening while we talk, or we'd translate ourselves. */
    private void beforeSound() {
        if (mode.equals("auto") && s.halfDuplex) {
            pauseMic();
        }
    }

    /** Wait for the current sound to end; false if a command stopped it (that command is kept unless it was STOP_SPEAKING). */
    private boolean waitPlaying() throws InterruptedException {
        while (player.isPlaying()) {
            Command c = commands.next(50);
            if (c != null) {
                player.stop();
                if (c.kind() != Kind.STOP_SPEAKING) {
                    commands.pushFront(c);
                }
                return false;
            }
        }
        return true;
    }

    private void play(AudioClip clip, String label) throws InterruptedException {
        if (clip == null || clip.isEmpty()) {
            return;
        }
        lastAudio = clip;
        beforeSound();
        ui.onState(State.SPEAKING, label);
        player.start(clip);
        waitPlaying();
    }

    /**
     * Speak {@code text} (already on screen) sentence by sentence: the next sentence is synthesized
     * while the previous one plays, so the voice starts sooner. Any command stops it.
     */
    void speak(String text, String lang) throws Exception {
        Tts tts = engine.tts;
        if (tts == null || text == null || text.isBlank()) {
            return;
        }
        List<String> parts = new ArrayList<>();
        for (String p : SENTENCE_END.split(text.strip())) {
            if (!p.isBlank()) {
                parts.add(p);
            }
        }
        if (parts.isEmpty()) {
            return;
        }
        double gapS = s.sentenceSilenceMs / 1000.0;
        List<AudioClip> clips = new ArrayList<>();
        clips.add(tts.synthesize(parts.get(0), lang));
        beforeSound();
        ui.onState(State.SPEAKING, name(lang));
        player.start(clips.get(0));
        boolean stopped = false;
        for (int i = 1; i < parts.size(); i++) {
            AudioClip next = tts.synthesize(parts.get(i), lang);   // while the previous sentence plays
            float[] padded = new float[(int) (gapS * next.sampleRate()) + next.samples().length];
            System.arraycopy(next.samples(), 0, padded, padded.length - next.samples().length, next.samples().length);
            next = new AudioClip(padded, next.sampleRate());
            clips.add(next);
            if (!waitPlaying()) {
                stopped = true;
                break;
            }
            player.start(next);
        }
        if (!stopped) {
            waitPlaying();
        }
        lastAudio = AudioClip.concat(clips, 0.0);
    }

    private void speakResult(TurnResult r) throws Exception {
        if (r.tgtLang.equals(mine) && !s.speakToMe) {
            return;
        }
        speak(r.tgtText, r.tgtLang);
    }

    private void replay() throws InterruptedException {
        if (lastAudio == null) {
            ui.onNote(Note.INFO, "nothing to replay yet");
            return;
        }
        play(lastAudio, "replay");
    }

    // ------------------------------------------------------------------ shared pieces

    private void show(TurnResult r, String who) {
        r.who = who;
        history.add(r);
        log.write(r);
        ui.onTurn(r);
    }

    private void toggle(String key) {
        boolean value;
        String label;
        if (key.equals("speak_to_me")) {
            value = s.speakToMe = !s.speakToMe;
            label = name(mine) + " audio";
        } else {
            value = s.confirmMySpeech = !s.confirmMySpeech;
            label = "confirm my speech";
        }
        ui.onSettingChanged(key, value);
        ui.onNote(Note.INFO, label + ": " + (value ? "on" : "off"));
    }

    private void level(float[] block, boolean speaking, double seconds) {
        long now = System.currentTimeMillis();
        if (now - lastLevelMs >= LEVEL_EVERY_MS) {
            lastLevelMs = now;
            ui.onLevel(Segmenter.levelDb(block), speaking, seconds);
        }
    }

    private void translateFile(String fileName, AudioClip clip16k) throws Exception {
        Segmenter seg = new Segmenter(vad, s.vad);
        List<Utterance> utts = new ArrayList<>(seg.feed(clip16k.samples()));
        Utterance last = seg.flush();
        if (last != null) {
            utts.add(last);
        }
        ui.onNote(Note.INFO, String.format(Locale.ROOT, "%s: %.0fs, %d part(s)", fileName, clip16k.seconds(), utts.size()));
        List<AudioClip> clips = new ArrayList<>();
        for (int i = 0; i < utts.size(); i++) {
            ui.onState(State.TRANSLATING, (i + 1) + "/" + utts.size());
            TurnResult r = engine.processAudio(utts.get(i).audio(), s.speakToMe, null);
            if (r != null && r.tgtText != null && !r.tgtText.isEmpty()) {
                show(r, utts.size() > 1 ? fileName + " · part " + (i + 1) : fileName);
                clips.add(r.audio);
            }
        }
        if (clips.isEmpty()) {
            ui.onNote(Note.WARNING, "no speech found in the file");
            return;
        }
        play(AudioClip.concat(clips, 0.4), "translation");
    }

    // ------------------------------------------------------------------ microphone (auto mode)

    private void startMic() throws Exception {
        if (!micOn) {
            mic.drain();   // stale audio from before the pause
            mic.start();
            micOn = true;
        }
    }

    private void stopMic() {
        if (micOn) {
            mic.stop();
            micOn = false;
        }
    }

    /** Stop listening until the current action is over (see runAuto). */
    private void pauseMic() {
        if (micOn) {
            stopMic();
            restartMic = true;
        }
    }

    // ------------------------------------------------------------------ auto mode

    private record Heard(float[] speech, Command command) {
    }

    /** Feed the microphone to the voice detector until a sentence is complete or a command arrives. */
    private Heard listen(Segmenter seg) throws InterruptedException {
        ui.onState(State.LISTENING, name(their) + " ⇄ " + name(mine));
        double speechS = 0;
        boolean wasSpeaking = false;
        while (true) {
            Command c = commands.next(0);
            if (c != null) {
                return new Heard(null, c);
            }
            float[] block = mic.read(50);
            if (block == null) {
                continue;
            }
            List<Utterance> utts = seg.feed(block);
            if (!utts.isEmpty()) {
                return new Heard(utts.get(0).audio(), null);
            }
            boolean speaking = seg.inSpeech();
            if (speaking != wasSpeaking) {
                ui.onState(speaking ? State.HEARING : State.LISTENING, speaking ? "" : name(their) + " ⇄ " + name(mine));
                wasSpeaking = speaking;
            }
            speechS = speaking ? speechS + block.length / (double) Segmenter.SAMPLE_RATE : 0.0;
            level(block, speaking, speechS);
        }
    }

    /** One sentence heard in auto mode: detect its language, translate, show, speak. */
    private void autoTurn(float[] audio) throws Exception {
        ui.onState(State.TRANSLATING, "");
        TurnResult r = engine.processAudio(audio, false, null);
        if (r == null || r.tgtText == null || r.tgtText.isEmpty()) {
            return;   // a cough, a door: stay quiet
        }
        lastHeardAudio = audio;
        lastHeardLang = r.srcLang;
        show(r, "");
        speakResult(r);
    }

    private void redoOtherLanguage() throws Exception {
        if (lastHeardAudio == null) {
            ui.onNote(Note.INFO, "nothing to redo yet");
            return;
        }
        String other = Languages.other(lastHeardLang, engine.languages);
        ui.onState(State.TRANSLATING, "again, as " + name(other));
        TurnResult r = engine.processAudio(lastHeardAudio, false, other);
        if (r == null || r.tgtText == null || r.tgtText.isEmpty()) {
            ui.onNote(Note.WARNING, "no " + name(other) + " words recognised in it");
            return;
        }
        lastHeardLang = other;
        show(r, "Redone");
        speakResult(r);
    }

    private void typeAny(String text, String lang) throws Exception {
        if (text == null || text.isBlank()) {
            return;
        }
        ui.onState(State.TRANSLATING, "");
        TurnResult r = engine.processText(text.strip(), lang, false);   // lang null: guessed from the words
        show(r, "Typed");
        speakResult(r);
    }

    /** Auto mode; returns "manual" or "quit". */
    String runAuto() throws InterruptedException {
        Segmenter seg = new Segmenter(vad, s.vad.withMinSilenceMs(s.autoEndSilenceMs));
        boolean paused = false;
        try {
            try {
                startMic();
            } catch (Exception e) {
                ui.onNote(Note.ERROR, "cannot open the microphone: " + e.getMessage());
                paused = true;
            }
            while (true) {
                Heard h;
                if (paused) {
                    ui.onState(State.PAUSED, "");
                    h = new Heard(null, commands.next(-1));
                } else {
                    h = listen(seg);
                }
                try {
                    if (h.speech() != null) {
                        autoTurn(h.speech());
                    } else {
                        Command c = h.command();
                        switch (c.kind()) {
                            case EXIT:
                                return "quit";
                            case SET_MODE:
                                if ("manual".equals(c.text())) {
                                    return "manual";
                                }
                                break;
                            case TOGGLE_PAUSE, PAUSE, RESUME: {
                                boolean pause = c.kind() == Kind.PAUSE || (c.kind() == Kind.TOGGLE_PAUSE && !paused);
                                if (pause && !paused) {
                                    paused = true;
                                    stopMic();
                                } else if (!pause && paused) {
                                    startMic();
                                    seg.reset();
                                    paused = false;
                                }
                                break;
                            }
                            case WRONG_LANGUAGE, TYPE, REPLAY, TOGGLE_SPEAK_TO_ME, TRANSLATE_FILE:
                                pauseMic();   // typing and replays aren't conversation
                                switch (c.kind()) {
                                    case WRONG_LANGUAGE -> redoOtherLanguage();
                                    case TYPE -> typeAny(c.text(), c.lang());
                                    case REPLAY -> replay();
                                    case TOGGLE_SPEAK_TO_ME -> toggle("speak_to_me");
                                    default -> translateFile(c.text(), c.clip());
                                }
                                break;
                            default:
                                break;   // manual-mode commands mean nothing here
                        }
                    }
                } catch (InterruptedException e) {
                    throw e;
                } catch (Exception e) {
                    error(e);
                }
                if (restartMic) {
                    restartMic = false;
                    if (!paused) {
                        Thread.sleep(s.playbackTailMs);   // let the room echo die down
                        try {
                            startMic();
                            seg.reset();
                        } catch (Exception e) {
                            ui.onNote(Note.ERROR, "cannot reopen the microphone: " + e.getMessage());
                            paused = true;
                        }
                    }
                }
            }
        } finally {
            stopMic();
            restartMic = false;
        }
    }

    // ------------------------------------------------------------------ manual mode

    private record Recording(float[] audio, String stoppedBy) {
    }

    /**
     * Record one utterance: from the first speech until {@code endSilenceMs} of silence. DONE stops
     * early and keeps what was said, even if the voice detector never triggered (quiet mic). CANCEL
     * gives up. Returns null when cancelled or when nobody spoke within the no-speech timeout.
     */
    private Recording recordUtterance(String lang, int endSilenceMs) throws Exception {
        VadParams p = s.vad.withMinSilenceMs(endSilenceMs).withMaxUtteranceS(s.maxRecordS);
        Segmenter seg = new Segmenter(vad, p);
        mic.drain();
        mic.start();
        ui.onState(State.RECORDING, name(lang));
        long t0 = System.nanoTime();
        boolean heard = false;
        List<float[]> raw = new ArrayList<>();
        int rawSamples = 0;
        try {
            while (true) {
                Command c = commands.next(0);
                if (c != null) {
                    switch (c.kind()) {
                        case CANCEL, SKIP:
                            return null;
                        case DONE, NEXT: {
                            Utterance u = seg.flush();
                            if (u != null) {
                                return new Recording(u.audio(), "key");
                            }
                            float[] audio = new float[rawSamples];
                            int pos = 0;
                            for (float[] b : raw) {
                                System.arraycopy(b, 0, audio, pos, b.length);
                                pos += b.length;
                            }
                            return audio.length > Segmenter.SAMPLE_RATE / 2 ? new Recording(audio, "key") : null;
                        }
                        case EXIT, SET_MODE:
                            commands.pushFront(c);
                            return null;
                        default:
                            break;
                    }
                }
                float[] block = mic.read(50);
                double elapsed = (System.nanoTime() - t0) / 1e9;
                if (block != null) {
                    if (elapsed <= s.maxRecordS) {
                        raw.add(block);
                        rawSamples += block.length;
                    }
                    for (Utterance u : seg.feed(block)) {
                        return new Recording(u.audio(), u.forcedCut() ? "limit" : "silence");
                    }
                    heard = heard || seg.inSpeech();
                    level(block, seg.inSpeech(), elapsed);
                }
                if (!heard && elapsed > s.noSpeechTimeoutS) {
                    return null;
                }
            }
        } finally {
            mic.stop();
        }
    }

    private Recording record(String lang, int endSilenceMs) throws Exception {
        Recording rec = recordUtterance(lang, endSilenceMs);
        if (rec == null) {
            ui.onNote(Note.INFO, "nothing recorded — if you did speak, watch the level bar; Done stops and keeps it");
        } else if (rec.stoppedBy().equals("limit")) {
            ui.onNote(Note.INFO, String.format(Locale.ROOT, "stopped at the %.0fs limit", s.maxRecordS));
        }
        return rec;
    }

    private String theirTurn() throws Exception {
        Recording rec = record(their, s.theirEndSilenceMs);
        if (rec == null) {
            return "ready";
        }
        ui.onState(State.TRANSLATING, "");
        TurnResult r = engine.processAudio(rec.audio(), false, their);
        if (r == null || r.tgtText == null || r.tgtText.isEmpty()) {
            ui.onNote(Note.WARNING, "Didn't catch any " + name(their) + " — listen again.");
            return "ready";
        }
        show(r, "They said");
        if (s.speakToMe) {
            speak(r.tgtText, r.tgtLang);
        }
        return "reply";
    }

    private String myTurn() throws Exception {
        while (true) {
            Recording rec = record(mine, s.myEndSilenceMs);
            if (rec == null) {
                return "reply";
            }
            ui.onState(State.TRANSLATING, "transcribing");
            long t0 = System.nanoTime();
            Transcript tr = engine.transcribe(rec.audio(), mine);
            double asrS = (System.nanoTime() - t0) / 1e9;
            if (tr == null) {
                ui.onNote(Note.WARNING, "Didn't catch that — try again, or type it.");
                return "reply";
            }
            String text = tr.text();
            if (s.confirmMySpeech) {
                text = confirm(tr.text());
                if (AGAIN.equals(text)) {
                    continue;
                }
                if (text == null) {
                    return "reply";
                }
            }
            return sendMine(text, asrS);
        }
    }

    /** Show what was heard; CONFIRM_OK accepts, CONFIRM_EDIT replaces, AGAIN re-records, CANCEL gives up. */
    private String confirm(String text) throws InterruptedException {
        ui.onState(State.CONFIRM, text);
        ui.onConfirm(text);
        while (true) {
            Command c = commands.next(-1);
            switch (c.kind()) {
                case CONFIRM_OK, NEXT, DONE:
                    return text;
                case CONFIRM_EDIT:
                    if (c.text() != null && !c.text().isBlank()) {
                        return c.text().strip();
                    }
                    break;
                case AGAIN:
                    return AGAIN;
                case CANCEL, SKIP:
                    ui.onNote(Note.INFO, "cancelled");
                    return null;
                case EXIT, SET_MODE:
                    commands.pushFront(c);
                    return null;
                default:
                    break;
            }
        }
    }

    private String sendMine(String text, double asrS) throws Exception {
        ui.onState(State.TRANSLATING, "into " + name(their));
        TurnResult r = engine.processText(text, mine, false);
        if (asrS >= 0) {
            java.util.Map<String, Double> t = new java.util.LinkedHashMap<>();
            t.put("asr", asrS);
            t.putAll(r.timings);
            r.timings = t;
        }
        show(r, "You said");
        speak(r.tgtText, r.tgtLang);
        return "ready";
    }

    private String typeText(String text, String lang) throws Exception {
        if (text == null || text.isBlank()) {
            return lang.equals(their) ? "ready" : "reply";
        }
        if (lang.equals(mine)) {
            return sendMine(text.strip(), -1);
        }
        ui.onState(State.TRANSLATING, "");
        TurnResult r = engine.processText(text.strip(), lang, false);
        show(r, "Typed");
        if (s.speakToMe) {
            speak(r.tgtText, r.tgtLang);
        }
        return "reply";
    }

    /** Manual mode; returns "auto" or "quit". */
    String runManual() throws InterruptedException {
        stopMic();
        String state = "ready";
        while (true) {
            ui.onState(state.equals("ready") ? State.READY : State.REPLY, "");
            Command c = commands.next(-1);
            try {
                state = step(state, c);
            } catch (InterruptedException e) {
                throw e;
            } catch (Exception e) {
                error(e);
                state = "ready";
            }
            if (state.equals("quit") || state.equals("auto")) {
                return state;
            }
        }
    }

    /** Handle one command in {@code state} ("ready" | "reply"); returns the next state. */
    String step(String state, Command c) throws Exception {
        switch (c.kind()) {
            case EXIT:
                return "quit";
            case SET_MODE:
                return "auto".equals(c.text()) ? "auto" : state;
            case NEXT:
                return state.equals("ready") ? theirTurn() : myTurn();
            case LISTEN_THEM:
                return theirTurn();
            case MY_TURN:
                return myTurn();
            case TYPE:
                return typeText(c.text(), c.lang() != null ? c.lang() : mine);
            case TRANSLATE_FILE:
                translateFile(c.text(), c.clip());
                return "ready";
            case SKIP, CANCEL:
                return "ready";
            case REPLAY:
                replay();
                return state;
            case TOGGLE_SPEAK_TO_ME:
                toggle("speak_to_me");
                return state;
            case TOGGLE_CONFIRM:
                toggle("confirm_my_speech");
                return state;
            default:
                return state;
        }
    }
}
