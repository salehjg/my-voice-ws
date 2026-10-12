package org.myvoice;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;
import org.myvoice.vad.EnergyVad;
import org.myvoice.vad.Segmenter;
import org.myvoice.vad.Utterance;
import org.myvoice.vad.VadParams;

/** Port of tests/test_audio_vad.py (segmenter part). */
public class SegmenterTest {
    static float[] tone(double seconds) {
        return Fakes.tone(seconds, 0.3f);
    }

    static float[] silence(double seconds) {
        return new float[(int) (seconds * Segmenter.SAMPLE_RATE)];
    }

    static Segmenter make(double maxUtteranceS) {
        return new Segmenter(new EnergyVad(-42.0), new VadParams(0.5f, 200, 400, 200, maxUtteranceS));
    }

    static List<Utterance> feedInBlocks(Segmenter seg, float[] audio) {
        List<Utterance> out = new ArrayList<>();
        for (int i = 0; i < audio.length; i += 700) {   // odd block size: exercises frame re-chunking
            float[] b = new float[Math.min(700, audio.length - i)];
            System.arraycopy(audio, i, b, 0, b.length);
            out.addAll(seg.feed(b));
        }
        return out;
    }

    @Test
    public void findsTwoUtterances() {
        float[] audio = Fakes.concat(silence(0.5), tone(1.0), silence(1.0), tone(0.8), silence(1.0));
        Segmenter seg = make(25);
        List<Utterance> utts = feedInBlocks(seg, audio);
        assertEquals(2, utts.size());
        Utterance first = utts.get(0), second = utts.get(1);
        assertTrue(first.startS() >= 0.25 && first.startS() <= 0.5);   // includes pre-roll
        assertTrue(first.durationS() >= 1.0 && first.durationS() <= 1.0 + 0.2 + 0.25 + 0.05);
        assertTrue(second.startS() >= 2.2 && second.startS() <= 2.5);
        assertFalse(seg.inSpeech());
    }

    @Test
    public void ignoresShortBlipsAndFlushesTail() {
        float[] audio = Fakes.concat(silence(0.3), tone(0.05), silence(0.8), tone(0.6));
        Segmenter seg = make(25);
        assertTrue(feedInBlocks(seg, audio).isEmpty());
        assertTrue(seg.inSpeech());
        Utterance last = seg.flush();
        assertNotNull(last);
        assertTrue(last.durationS() >= 0.6);
    }

    @Test
    public void forcesCutOnLongSpeech() {
        List<Utterance> utts = feedInBlocks(make(1.0), tone(2.5));
        assertEquals(2, utts.size());
        assertTrue(utts.get(0).forcedCut() && utts.get(1).forcedCut());
    }

    @Test
    public void resetDropsPartialSpeech() {
        Segmenter seg = make(25);
        feedInBlocks(seg, tone(0.5));
        assertTrue(seg.inSpeech());
        seg.reset();
        assertFalse(seg.inSpeech());
        assertNull(seg.flush());
    }
}
