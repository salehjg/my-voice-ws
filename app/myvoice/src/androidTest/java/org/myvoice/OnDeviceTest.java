package org.myvoice;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.io.InputStream;

import android.content.Context;
import android.util.Log;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.myvoice.accel.Choice;
import org.myvoice.asr.Asr;
import org.myvoice.asr.Transcript;
import org.myvoice.audio.AudioClip;
import org.myvoice.audio.Wav;
import org.myvoice.core.EngineFactory;
import org.myvoice.models.ModelCatalog.Model;
import org.myvoice.models.ModelStore;
import org.myvoice.mt.Translator;
import org.myvoice.nativebridge.NativeLib;
import org.myvoice.vad.SileroVad;

/**
 * Runs on the phone (./gradlew connectedDebugAndroidTest). Native-library checks always run;
 * model checks run for the models that are installed. Results go to logcat (tag "myvoice-test").
 */
@RunWith(AndroidJUnit4.class)
public class OnDeviceTest {
    private static final String TAG = "myvoice-test";

    private Context ctx() {
        return InstrumentationRegistry.getInstrumentation().getTargetContext();
    }

    private AudioClip probeClip() throws Exception {
        try (InputStream in = ctx().getAssets().open("probe_it.wav")) {
            return Wav.read(in).resampled(16000);
        }
    }

    @Test
    public void nativeLibraryLoadsTheRightCpuVariant() {
        assertTrue(NativeLib.loadError(), NativeLib.loaded());
        String[] devices = EngineFactory.loadGgml(ctx(), "");
        Log.i(TAG, "ggml devices: " + String.join(" | ", devices) + "; CPU features: " + NativeLib.ggmlCpuFeatures());
        assertTrue(devices.length >= 1);
        assertTrue(devices[0], devices[0].contains("\tcpu\t"));
    }

    @Test
    public void espeakGivesTheLaptopsPhonemes() throws Exception {
        ModelStore store = new ModelStore(ctx());
        java.io.File data = store.extractAssets("espeak-ng-data", EngineFactory.ESPEAK_DATA_VERSION);
        assertEquals(0, NativeLib.espeakInit(data.getParent()));
        assertEquals(0, NativeLib.espeakSetVoice("it"));
        // reference: the laptop build of the same code, identical to Piper's own espeak bridge
        assertArrayEquals(new String[] {"bʊondʒˈɔɾno\u0001,\u00010", "kˌome stˈa\u0001?\u00011"},
                NativeLib.espeakPhonemize("Buongiorno, come sta?"));
        assertEquals(0, NativeLib.espeakSetVoice("en-us"));
        assertArrayEquals(new String[] {"həlˈoʊ ðˈɛɹ\u0001,\u00010", "hˈaʊ ɑːɹ juː\u0001?\u00011"},
                NativeLib.espeakPhonemize("Hello there, how are you?"));
    }

    @Test
    public void sileroHearsTheProbeClip() throws Exception {
        try (SileroVad vad = new EngineFactory(ctx(), new ModelStore(ctx())).vad()) {
            float[] s = probeClip().samples();
            int speech = 0, frames = 0;
            for (int i = 0; i + 512 <= s.length; i += 512, frames++) {
                float[] f = new float[512];
                System.arraycopy(s, i, f, 0, 512);
                if (vad.prob(f) > 0.5f) {
                    speech++;
                }
            }
            Log.i(TAG, "silero: " + speech + "/" + frames + " speech frames");
            assertTrue(speech > frames / 2);
        }
    }

    private Model installed(String component) {
        ModelStore store = new ModelStore(ctx());
        for (Model m : store.catalog().component(component)) {
            if (store.installed(m)) {
                return m;
            }
        }
        return null;
    }

    @Test
    public void whisperTranscribesOnTheCpu() throws Exception {
        Model m = installed("asr");
        assumeTrue("no Whisper model installed", m != null);
        EngineFactory f = new EngineFactory(ctx(), new ModelStore(ctx()));
        try (Asr asr = f.asr(m, Choice.cpu(4, ""), "", true)) {
            long t0 = System.nanoTime();
            Transcript tr = asr.transcribe(probeClip().samples(), EngineFactory.languages(), null);
            Log.i(TAG, m.id() + ": " + tr + " in " + (System.nanoTime() - t0) / 1e9 + " s");
            assertNotNull(tr);
            assertEquals("it", tr.language());
            assertTrue(tr.text(), tr.text().toLowerCase().contains("caff"));
        }
    }

    @Test
    public void translationWorksOnTheCpu() throws Exception {
        Model m = installed("mt");
        assumeTrue("no translation model installed", m != null);
        EngineFactory f = new EngineFactory(ctx(), new ModelStore(ctx()));
        try (Translator mt = f.translator(m, Choice.cpu(4, ""))) {
            long t0 = System.nanoTime();
            String out = mt.translate("Buongiorno, vorrei prenotare un tavolo per due persone stasera alle otto.", "it", "en");
            Log.i(TAG, m.id() + ": " + out + " in " + (System.nanoTime() - t0) / 1e9 + " s");
            assertTrue(out, out.toLowerCase().contains("table"));
        }
    }
}
