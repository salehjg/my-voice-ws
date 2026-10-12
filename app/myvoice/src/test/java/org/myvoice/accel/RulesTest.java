package org.myvoice.accel;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** Small rules: probe output comparison, Whisper's shortened window and repetition check. */
public class RulesTest {
    @Test
    public void probeOutputsAreComparedSensibly() {
        assertEquals(1.0, AccelerationManager.similarity("Good morning, a table for two.", "good morning a table for two", "mt"), 1e-9);
        assertTrue(AccelerationManager.similarity("Good morning, a table for two.", "Good morning, the table for two.", "mt") >= 0.6);
        assertTrue(AccelerationManager.similarity("Good morning, a table for two.", "<unk> <unk> <unk>", "mt") < 0.6);
        assertTrue(AccelerationManager.similarity("2.1s rms 0.120", "2.0s rms 0.118", "tts") > 0.9);
        assertTrue(AccelerationManager.similarity("2.1s rms 0.120", "0.2s rms 0.001", "tts") < 0.6);
    }

    @Test
    public void whisperWindowIsNeverTooShort() {
        // measured: a 3.9 s Italian clip repeats itself below ~450 frames (laptop); on the phone only
        // multiples of 256 are fast (512: 1.1 s, 514: 16 s)
        assertEquals(512, org.myvoice.asr.WhisperCppAsrRules.audioCtxFor((int) (3.85 * 16000)));
        assertEquals(512, org.myvoice.asr.WhisperCppAsrRules.audioCtxFor(16000));   // the floor
        assertEquals(768, org.myvoice.asr.WhisperCppAsrRules.audioCtxFor(5 * 16000));
        assertEquals(1024, org.myvoice.asr.WhisperCppAsrRules.audioCtxFor(6700 * 16));
        assertEquals(1280, org.myvoice.asr.WhisperCppAsrRules.audioCtxFor(10 * 16000));
        assertEquals(1500, org.myvoice.asr.WhisperCppAsrRules.audioCtxFor(30 * 16000));
        for (int ms = 200; ms <= 30000; ms += 100) {
            int ctx = org.myvoice.asr.WhisperCppAsrRules.audioCtxFor(ms * 16);
            int frames = (int) Math.ceil(ms / 1000.0 * 50);
            assertTrue(ms + " ms", ctx == 1500 || (ctx % 256 == 0 && ctx >= 2.3 * frames));
        }
    }

    @Test
    public void repeatedSentencesAreDropped() {
        assertEquals("Buongiorno vorrei un caffè per favore.", org.myvoice.asr.WhisperCppAsrRules.dropRepeatedSentences(
                " Buongiorno vorrei un caffè per favore. Buongiorno vorrei un caffè per favore."));
        assertEquals("Sì. Certo. Sì.", org.myvoice.asr.WhisperCppAsrRules.dropRepeatedSentences("Sì. Certo. Sì."));
        assertEquals("Ciao!", org.myvoice.asr.WhisperCppAsrRules.dropRepeatedSentences("Ciao! ciao!"));
    }

    @Test
    public void repetitionIsSpotted() {
        String loop = "Buongiorno vorrei un caffè e un cornetto per favore. ".repeat(5);
        assertTrue(org.myvoice.asr.WhisperCppAsrRules.looksRepeated(loop, (int) (3.85 * 16000)));
        assertFalse(org.myvoice.asr.WhisperCppAsrRules.looksRepeated("Buongiorno vorrei un caffè e un cornetto per favore.",
                (int) (3.85 * 16000)));
    }
}
