package org.myvoice.service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import android.content.Context;

import org.myvoice.accel.AccelerationManager;
import org.myvoice.accel.Backend;
import org.myvoice.accel.Choice;
import org.myvoice.accel.ProbeResult;
import org.myvoice.accel.Stage;
import org.myvoice.asr.Asr;
import org.myvoice.core.Engine;
import org.myvoice.core.EngineFactory;
import org.myvoice.core.Resilient;
import org.myvoice.core.Settings;
import org.myvoice.core.SettingsStore;
import org.myvoice.models.ModelCatalog;
import org.myvoice.models.ModelCatalog.Model;
import org.myvoice.models.ModelStore;
import org.myvoice.mt.Translator;
import org.myvoice.tts.RoutingTts;
import org.myvoice.tts.SystemTts;
import org.myvoice.tts.Tts;

/** Builds the Engine from the chosen models, on the backends the probe picked (with fallbacks). */
final class EngineLoader {
    private final Context context;
    private final ModelStore store;
    private final ModelCatalog catalog;
    private final AccelerationManager accel;
    private final EngineFactory factory;
    final Map<Stage, String> status = new HashMap<>();   // stage -> what it runs on, for the chips

    EngineLoader(Context c, AccelerationManager accel) {
        context = c.getApplicationContext();
        store = new ModelStore(context);
        catalog = store.catalog();
        this.accel = accel;
        factory = new EngineFactory(context, store);
    }

    static String modelPref(Context c, String component, ModelCatalog cat) {
        String id = SettingsStore.get(c, SettingsStore.MODEL_PREFIX + component, "");
        if (!id.isEmpty() && cat.get(id) != null) {
            return id;
        }
        ModelCatalog.Preset p = cat.preset(cat.defaultPreset);
        for (String m : p.models()) {
            Model x = cat.get(m);
            if (x != null && x.component().equals(component.startsWith("voice") ? "voice" : component)
                    && (!component.startsWith("voice_") || x.lang().equals(component.substring(6)))) {
                return m;
            }
        }
        return "";
    }

    /** The chosen backend first, then the other ones that worked in the probe (fastest first), the CPU last. */
    private List<Choice> chain(Stage stage, Model m, String prefKey) {
        Choice first = accel.choose(stage, m.id(), SettingsStore.get(context, prefKey, "auto"));
        List<Choice> chain = new ArrayList<>();
        chain.add(first);
        List<ProbeResult> rs = new ArrayList<>(accel.results(stage, m.id()));
        rs.sort(Comparator.comparingDouble(ProbeResult::seconds));
        for (ProbeResult r : rs) {
            if (r.usable() && r.backend() != Backend.CPU && !(r.backend() == first.backend() && r.device().equals(first.device()))) {
                chain.add(new Choice(r.backend(), r.device(), accel.label(r.device()), r.threads(), "fallback"));
            }
        }
        if (first.backend() != Backend.CPU) {
            chain.add(Choice.cpu(AccelerationManager.defaultThreads(), "fallback"));
        }
        return chain;
    }

    Engine load(Settings s, Consumer<String> progress, Resilient.OnSwitch onSwitch) throws Exception {
        Model asrM = catalog.get(modelPref(context, "asr", catalog));
        Model mtM = catalog.get(modelPref(context, "mt", catalog));
        if (!store.installed(asrM) || !store.installed(mtM)) {
            throw new IllegalStateException("models are missing: open Settings → Models");
        }
        boolean shortCtx = SettingsStore.prefs(context).getBoolean(SettingsStore.SHORT_AUDIO_CTX, true);

        progress.accept("Loading " + asrM.name() + "…");
        Asr asr = Resilient.asr(Stage.ASR, chain(Stage.ASR, asrM, SettingsStore.ACCEL_PREFIX + "asr"),
                c -> factory.asr(asrM, c, s.initialPrompt, shortCtx), onSwitch);
        status.put(Stage.ASR, asr.describe());

        progress.accept("Loading " + mtM.name() + "…");
        Stage mtStage = mtM.engine().equals("marian") ? Stage.MT_ONNX : Stage.MT_LLM;
        Translator mt;
        try {
            mt = Resilient.translator(mtStage, chain(mtStage, mtM, SettingsStore.ACCEL_PREFIX + "mt"),
                    c -> factory.translator(mtM, c), onSwitch);
        } catch (Exception e) {
            asr.close();
            throw e;
        }
        status.put(mtStage, mt.describe());

        progress.accept("Loading voices…");
        Tts tts;
        try {
            tts = voices(s, onSwitch);
        } catch (Exception e) {
            asr.close();
            mt.close();
            throw e;
        }
        status.put(Stage.TTS_SYSTEM, tts.describe());
        return new Engine(s, asr, mt, tts);
    }

    /** Each language: the phone's voice ("system:...") or a Piper voice ("piper:<model>"); default: phone voice if it has one offline. */
    private Tts voices(Settings s, Resilient.OnSwitch onSwitch) throws Exception {
        Map<String, String> pref = new HashMap<>();
        for (String lang : s.languages) {
            pref.put(lang, SettingsStore.get(context, SettingsStore.VOICE_PREFIX + lang, ""));
        }
        SystemTts system = null;
        try {
            Map<String, String> sysVoices = new HashMap<>();
            for (String lang : s.languages) {
                String v = pref.get(lang);
                sysVoices.put(lang, v.startsWith(SystemTts.PREFIX) ? v : "");
            }
            system = new SystemTts(context, sysVoices);
        } catch (Exception e) {
            system = null;   // no system engine: Piper for everything
        }
        Map<String, Model> piperVoices = new HashMap<>();
        Map<String, Tts> byLang = new HashMap<>();
        for (String lang : s.languages) {
            String v = pref.get(lang);
            boolean wantPiper = v.startsWith("piper:");
            if (!wantPiper && system != null && !system.voices(lang).isEmpty()) {
                byLang.put(lang, system);
                continue;
            }
            Model m = wantPiper ? catalog.get(v.substring(6)) : null;
            if (m == null || !store.installed(m)) {
                m = catalog.get(modelPref(context, "voice_" + lang, catalog));
            }
            if (m == null || !store.installed(m)) {
                for (Model x : catalog.voices(lang)) {
                    if (store.installed(x)) {
                        m = x;
                        break;
                    }
                }
            }
            if (m != null && store.installed(m)) {
                piperVoices.put(lang, m);
            } else if (system != null) {
                byLang.put(lang, system);   // last resort: the engine's default voice
            } else {
                throw new IllegalStateException("no voice for " + lang + ": install a Piper voice or the phone's voice data");
            }
        }
        if (!piperVoices.isEmpty()) {
            Model any = piperVoices.values().iterator().next();
            Tts piper = Resilient.tts(Stage.TTS_PIPER, chain(Stage.TTS_PIPER, any, SettingsStore.ACCEL_PREFIX + "tts"),
                    c -> factory.piper(piperVoices, c, s.sentenceSilenceMs), onSwitch);
            for (String lang : piperVoices.keySet()) {
                byLang.put(lang, piper);
            }
        }
        if (system != null && !byLang.containsValue(system)) {
            system.close();
        }
        return new RoutingTts(byLang);
    }
}
