package org.myvoice;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;
import org.myvoice.audio.AudioClip;
import org.myvoice.core.Command;
import org.myvoice.core.Command.Kind;
import org.myvoice.core.Commands;
import org.myvoice.core.Conversation;
import org.myvoice.core.ConversationListener.State;
import org.myvoice.core.Engine;
import org.myvoice.core.SessionLog;
import org.myvoice.core.Settings;
import org.myvoice.vad.EnergyVad;

/** Port of tests/test_talk.py: both modes with fake models, microphone and speaker. */
public class ConversationTest {
    static Command cmd(Kind k) {
        return Command.of(k);
    }

    /** Everything one test needs. */
    static final class Rig {
        final Settings settings = new Settings();
        Fakes.FakeAsr asr = new Fakes.FakeAsr();
        final Fakes.FakeMt mt = new Fakes.FakeMt();
        final Fakes.FakeTts tts = new Fakes.FakeTts();
        final Fakes.Player player = new Fakes.Player();
        final Fakes.Recorder ui = new Fakes.Recorder();
        Fakes.Mic mic;
        Commands commands;
        Conversation conv;

        Rig() {
            settings.playbackTailMs = 0;
        }

        Rig run(String mode) throws InterruptedException {
            Engine engine = new Engine(settings, asr, mt, tts);
            conv = new Conversation(engine, settings, commands, ui, player, mic, new EnergyVad(-42.0), new SessionLog(null, ""));
            conv.run(mode);
            return this;
        }

        List<String> mtCalls() {
            List<String> out = new ArrayList<>();
            for (String[] c : mt.calls) {
                out.add(c[0] + "|" + c[1] + ">" + c[2]);
            }
            return out;
        }

        List<String> spoken() {
            List<String> out = new ArrayList<>();
            for (String[] s : tts.spoken) {
                out.add(s[0] + ":" + s[1]);
            }
            return out;
        }
    }

    static Rig manual(List<Command> cmds, float[]... clips) {
        Rig r = new Rig();
        r.mic = new Fakes.Mic(clips.length > 0 ? List.of(clips) : List.of(Fakes.tone(0.8)), true);
        r.commands = new Fakes.Scripted(cmds);
        return r;
    }

    static Rig auto(List<Command> cmds, List<String> detect, float prob, float[]... clips) {
        Rig r = new Rig();
        r.asr = new Fakes.FakeAsr(detect, prob);
        r.mic = new Fakes.Mic(List.of(clips), false);
        r.commands = new Fakes.WhenMicIdle(cmds, r.mic);
        return r;
    }

    // ------------------------------------------------------------------ manual mode

    @Test
    public void fullCycleListenTranslateAnswerSpeak() throws Exception {
        // NEXT: listen to Italian -> English; NEXT: record English; OK -> spoken Italian
        Rig r = manual(List.of(cmd(Kind.NEXT), cmd(Kind.NEXT), cmd(Kind.CONFIRM_OK), cmd(Kind.EXIT)),
                Fakes.tone(0.8), Fakes.tone(1.6)).run("manual");
        assertEquals(2, r.mic.n);   // one mic, started once per recording
        assertEquals(List.of("it", "en"), r.asr.calls);   // languages forced per step, no detection
        assertEquals(List.of("Buongiorno, come sta?|it>en", "Very well, thank you.|en>it"), r.mtCalls());
        assertEquals(List.of("en:<en> Buongiorno, come sta?", "it:<it> Very well, thank you."), r.spoken());
        assertEquals(2, r.player.played.size());
        assertEquals("They said", r.ui.turns.get(0).who);
        assertEquals("You said", r.ui.turns.get(1).who);
        assertTrue(r.ui.states.contains(State.REPLY));
        assertEquals(List.of("Very well, thank you."), r.ui.confirms);
    }

    @Test
    public void englishAudioOffOnlyShows() throws Exception {
        Rig r = manual(List.of(cmd(Kind.NEXT), cmd(Kind.SKIP), cmd(Kind.EXIT)));
        r.settings.speakToMe = false;
        r.run("manual");
        assertTrue(r.tts.spoken.isEmpty());
        assertTrue(r.player.played.isEmpty());
        assertEquals(List.of("Buongiorno, come sta?|it>en"), r.mtCalls());
    }

    @Test
    public void confirmEditAndRecordAgain() throws Exception {
        // MY_TURN: answer directly; AGAIN: record again; EDIT: the translated text is the edited one
        Rig r = manual(List.of(cmd(Kind.MY_TURN), cmd(Kind.AGAIN), Command.edit("Very well, and you?"), cmd(Kind.EXIT)))
                .run("manual");
        assertEquals(2, r.mic.starts);
        assertEquals(List.of("Very well, and you?|en>it"), r.mtCalls());
    }

    @Test
    public void confirmCancelReturnsToYourTurn() throws Exception {
        Rig r = manual(List.of(cmd(Kind.MY_TURN), cmd(Kind.CANCEL), cmd(Kind.EXIT))).run("manual");
        assertTrue(r.mt.calls.isEmpty());
        assertTrue(r.ui.allNotes().contains("cancelled"));
    }

    @Test
    public void noConfirmTranslatesImmediately() throws Exception {
        Rig r = manual(List.of(cmd(Kind.MY_TURN), cmd(Kind.EXIT)));
        r.settings.confirmMySpeech = false;
        r.run("manual");
        assertEquals(List.of("Very well, thank you.|en>it"), r.mtCalls());
    }

    @Test
    public void typeBothLanguagesAndReplay() throws Exception {
        Rig r = manual(List.of(Command.type("See you tomorrow", "en"), Command.type("A domani", "it"),
                cmd(Kind.REPLAY), cmd(Kind.EXIT))).run("manual");
        assertEquals(List.of("See you tomorrow|en>it", "A domani|it>en"), r.mtCalls());
        assertEquals(3, r.player.played.size());   // Italian speech, English speech, replay
    }

    @Test
    public void togglesChangeSettings() throws Exception {
        Rig r = manual(List.of(cmd(Kind.TOGGLE_SPEAK_TO_ME), cmd(Kind.TOGGLE_CONFIRM), cmd(Kind.EXIT))).run("manual");
        assertFalse(r.settings.speakToMe);
        assertFalse(r.settings.confirmMySpeech);
        assertEquals(Boolean.FALSE, r.ui.settings.get("speak_to_me"));
        assertEquals(Boolean.FALSE, r.ui.settings.get("confirm_my_speech"));
    }

    @Test
    public void translateFile() throws Exception {
        AudioClip note = new AudioClip(Fakes.concat(Fakes.silence(0.5), Fakes.tone(0.8), Fakes.silence(1.0)), 16000);
        Rig r = manual(List.of(Command.file("note.wav", note), cmd(Kind.EXIT))).run("manual");
        assertEquals(List.of("Buongiorno, come sta?|it>en"), r.mtCalls());
        assertEquals("note.wav", r.ui.turns.get(0).who);
        assertTrue(r.ui.allNotes().contains("note.wav"));
    }

    @Test
    public void manualStopKeepsAudioTheVadDidNotHear() throws Exception {
        Rig r = new Rig();
        r.mic = new Fakes.QuietMic();
        r.commands = new Fakes.OnPoll(40, cmd(Kind.DONE), List.of(cmd(Kind.MY_TURN), cmd(Kind.CONFIRM_OK), cmd(Kind.EXIT)));
        r.run("manual");
        assertEquals(List.of("en"), r.asr.calls);   // the quiet recording was kept and transcribed
        assertEquals(List.of("Very well, thank you.|en>it"), r.mtCalls());
    }

    @Test
    public void cancelDuringRecording() throws Exception {
        Rig r = new Rig();
        r.mic = new Fakes.Mic(List.of(Fakes.tone(0.8)), true);
        r.commands = new Fakes.OnPoll(5, cmd(Kind.CANCEL), List.of(cmd(Kind.LISTEN_THEM), cmd(Kind.EXIT)));
        r.run("manual");
        assertTrue(r.asr.calls.isEmpty());
        assertTrue(r.ui.allNotes().contains("nothing recorded"));
    }

    @Test
    public void multiSentenceReplyIsSpokenSentenceBySentence() throws Exception {
        Rig r = manual(List.of(Command.type("It costs six euros. The bus leaves at nine! Have a good trip.", "en"),
                cmd(Kind.REPLAY), cmd(Kind.EXIT))).run("manual");
        List<String> it = new ArrayList<>();
        for (String[] s : r.tts.spoken) {
            if (s[0].equals("it")) {
                it.add(s[1]);
            }
        }
        assertEquals(List.of("<it> It costs six euros.", "The bus leaves at nine!", "Have a good trip."), it);
        assertEquals(4, r.player.played.size());   // three sentences + one replay of the joined audio
        assertTrue(r.player.played.get(3) > 3 * 2205);
    }

    @Test
    public void errorsAreShownAndTheAppKeepsGoing() throws Exception {
        Rig r = manual(List.of(Command.type("Hello", "en"), cmd(Kind.NEXT), cmd(Kind.EXIT)));
        r.tts.fail = new IllegalStateException("voice exploded");
        r.run("manual");
        long n = r.ui.notes.stream().filter(s -> s.contains("voice exploded")).count();
        assertEquals(2, n);   // typed answer + their next turn, both survived
        assertEquals(List.of("Hello|en>it", "Buongiorno, come sta?|it>en"), r.mtCalls());
    }

    // ------------------------------------------------------------------ auto mode

    @Test
    public void autoModeTranslatesWhateverItHears() throws Exception {
        // Italian -> spoken in English; mic paused while speaking; English -> spoken in Italian
        Rig r = auto(List.of(cmd(Kind.EXIT)), List.of("it", "en"), 0.99f, Fakes.tone(0.8), Fakes.tone(1.6)).run("auto");
        assertEquals(List.of("null", "null"), r.asr.calls);   // language detected, not forced
        assertEquals(List.of("Buongiorno, come sta?|it>en", "Very well, thank you.|en>it"), r.mtCalls());
        assertEquals(List.of("en:<en> Buongiorno, come sta?", "it:<it> Very well, thank you."), r.spoken());
        assertEquals(2, r.player.played.size());
        assertEquals(3, r.mic.starts);   // restarted after each translation
        assertEquals(3, r.mic.stops);    // ...and stopped on exit
        assertEquals(List.of("auto"), r.ui.modes);
        assertEquals("", r.ui.turns.get(0).who);
    }

    @Test
    public void autoModeWithHeadphonesKeepsListening() throws Exception {
        Rig r = auto(List.of(cmd(Kind.EXIT)), List.of("it", "en"), 0.99f,
                Fakes.concat(Fakes.tone(0.8), Fakes.silence(2), Fakes.tone(1.6)));
        r.settings.halfDuplex = false;
        r.run("auto");
        assertEquals(2, r.mt.calls.size());
        assertEquals(2, r.player.played.size());
        assertEquals(1, r.mic.starts);   // never paused
    }

    @Test
    public void autoModeMyAudioOffOnlySpeaksTheirLanguage() throws Exception {
        Rig r = auto(List.of(cmd(Kind.EXIT)), List.of("it", "en"), 0.99f,
                Fakes.concat(Fakes.tone(0.8), Fakes.silence(2), Fakes.tone(1.6)));
        r.settings.speakToMe = false;
        r.run("auto");
        assertEquals(2, r.mt.calls.size());
        assertEquals(List.of("it:<it> Very well, thank you."), r.spoken());
    }

    @Test
    public void autoModeWrongLanguageRedo() throws Exception {
        Rig r = auto(List.of(cmd(Kind.WRONG_LANGUAGE), cmd(Kind.EXIT)), List.of("it"), 0.6f, Fakes.tone(0.8)).run("auto");
        assertEquals(List.of("null", "en"), r.asr.calls);   // redone as the other language
        assertArrayEquals(new String[] {"Very well, thank you.", "en", "it"}, r.mt.calls.get(1));
        assertEquals(0.6f, r.ui.turns.get(0).langProb, 1e-6);
        assertEquals("Redone", r.ui.turns.get(1).who);
    }

    @Test
    public void autoModeTypeAndReplay() throws Exception {
        Rig r = auto(List.of(Command.type("Ciao a tutti", null), cmd(Kind.REPLAY), cmd(Kind.EXIT)), List.of(), 0.99f,
                new float[100]).run("auto");
        assertEquals(List.of("Ciao a tutti|it>en"), r.mtCalls());   // language guessed from the words
        assertEquals(2, r.player.played.size());   // spoken + replay
    }

    @Test
    public void pauseThenSwitchModesBothWays() throws Exception {
        Rig r = auto(List.of(cmd(Kind.TOGGLE_PAUSE), Command.mode("manual"), Command.mode("auto"), cmd(Kind.EXIT)),
                List.of(), 0.99f, new float[100]).run("auto");
        assertTrue(r.ui.states.contains(State.PAUSED));
        assertEquals(List.of("auto", "manual", "auto"), r.ui.modes);
        assertEquals(2, r.mic.starts);   // the mic is off in manual mode until you record
        assertEquals(2, r.mic.stops);
        assertTrue(r.mt.calls.isEmpty());
        assertEquals(State.STOPPED, r.ui.states.get(r.ui.states.size() - 1));
    }
}
