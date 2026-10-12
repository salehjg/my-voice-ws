package org.myvoice.core;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import android.content.Context;

import org.myvoice.accel.Backend;
import org.myvoice.accel.Choice;
import org.myvoice.asr.Asr;
import org.myvoice.asr.WhisperCppAsr;
import org.myvoice.models.ModelCatalog.FileSpec;
import org.myvoice.models.ModelCatalog.Model;
import org.myvoice.models.ModelStore;
import org.myvoice.mt.LlamaTranslator;
import org.myvoice.mt.MarianTranslator;
import org.myvoice.mt.Translator;
import org.myvoice.nativebridge.NativeLib;
import org.myvoice.tts.EspeakPhonemizer;
import org.myvoice.tts.PiperTts;
import org.myvoice.vad.SileroVad;

/** Builds each engine from a catalogue model on a chosen backend (used by the service and the probe). */
public final class EngineFactory {
    public static final int ESPEAK_DATA_VERSION = 1;
    private final Context context;
    private final ModelStore store;
    private static boolean backendsLoaded;
    private static String backendsLoadedFor = "";

    public EngineFactory(Context context, ModelStore store) {
        this.context = context.getApplicationContext();
        this.store = store;
    }

    /**
     * Load ggml's backend modules: the best CPU variant always, GPU ones only when asked (their
     * drivers are only touched after the probe process has shown they don't crash).
     */
    public static synchronized String[] loadGgml(Context c, String gpuBackends) {
        if (!NativeLib.loaded()) {
            throw new IllegalStateException("native library missing: " + NativeLib.loadError());
        }
        String[] devices = NativeLib.ggmlLoadBackends(c.getApplicationInfo().nativeLibraryDir, gpuBackends);
        backendsLoaded = true;
        backendsLoadedFor = gpuBackends;
        return devices;
    }

    private void ensureGgml(Choice choice) {
        String want = switch (choice.backend()) {
            case VULKAN -> "vulkan";
            case OPENCL -> "opencl";
            default -> "";
        };
        if (!backendsLoaded || (!want.isEmpty() && !backendsLoadedFor.contains(want))) {
            loadGgml(context, backendsLoadedFor.isEmpty() ? want : backendsLoadedFor + (want.isEmpty() ? "" : "," + want));
        }
    }

    private File first(Model m) {
        return store.file(m.files().get(0));
    }

    public Asr asr(Model m, Choice c, String initialPrompt, boolean shortAudioCtx) {
        ensureGgml(c);
        return new WhisperCppAsr(first(m), c.backend(), c.device(), c.deviceLabel(), c.threads(), initialPrompt, shortAudioCtx);
    }

    public Translator translator(Model m, Choice c) throws Exception {
        if (m.engine().equals("marian")) {
            Map<String, File> dirs = new HashMap<>();
            for (FileSpec f : m.files()) {
                String[] parts = f.path().split("/");   // opus/<direction>/...
                dirs.put(parts[1], store.dir(parts[0] + "/" + parts[1]));
            }
            return new MarianTranslator(dirs, c.backend(), c.threads());
        }
        ensureGgml(c);
        LlamaTranslator.Style style = "chat".equals(m.style()) ? LlamaTranslator.Style.CHAT : LlamaTranslator.Style.TRANSLATEGEMMA;
        return new LlamaTranslator(first(m), style, c.backend(), c.device(), c.deviceLabel(), c.threads());
    }

    /** Piper voices for the given languages (language -> catalogue voice). */
    public PiperTts piper(Map<String, Model> voices, Choice c, int sentenceSilenceMs) throws Exception {
        File data = store.extractAssets("espeak-ng-data", ESPEAK_DATA_VERSION);
        EspeakPhonemizer.init(data.getParent());
        Map<String, File> files = new HashMap<>();
        for (Map.Entry<String, Model> e : voices.entrySet()) {
            for (FileSpec f : e.getValue().files()) {
                if (f.path().endsWith(".onnx")) {
                    files.put(e.getKey(), store.file(f));
                }
            }
        }
        return new PiperTts(files, c.backend() == Backend.SYSTEM ? Backend.CPU : c.backend(), c.threads(), sentenceSilenceMs);
    }

    public SileroVad vad() throws Exception {
        return new SileroVad(store.asset("silero_vad.onnx"));
    }

    public static List<String> languages() {
        return List.of("it", "en");
    }
}
