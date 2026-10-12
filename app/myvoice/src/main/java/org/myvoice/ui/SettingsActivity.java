package org.myvoice.ui;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import android.content.Intent;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.speech.tts.Voice;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.preference.ListPreference;
import androidx.preference.Preference;
import androidx.preference.PreferenceFragmentCompat;

import org.myvoice.R;
import org.myvoice.audio.AudioClip;
import org.myvoice.audio.AudioTrackPlayer;
import org.myvoice.core.EngineFactory;
import org.myvoice.core.SettingsStore;
import org.myvoice.models.ModelCatalog.Model;
import org.myvoice.models.ModelStore;
import org.myvoice.service.TranslatorService;
import org.myvoice.tts.PiperTts;
import org.myvoice.tts.SystemTts;

/** Voices, behaviour, listening, models and acceleration. Changes apply the next time the conversation starts. */
public final class SettingsActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        if (saved == null) {
            getSupportFragmentManager().beginTransaction().replace(android.R.id.content, new Fragment()).commit();
        }
    }

    public static final class Fragment extends PreferenceFragmentCompat {
        private static final Map<String, String> SAMPLES = Map.of(
                "it", "Ciao! Questa è la voce italiana. Come ti sembra?",
                "en", "Hi! This is the English voice. How does it sound?");
        private TextToSpeech tts;

        @Override
        public void onCreatePreferences(Bundle saved, String rootKey) {
            setPreferencesFromResource(R.xml.preferences, rootKey);
            click("models", () -> startActivity(new Intent(requireContext(), SetupActivity.class)));
            click("acceleration_report", () -> startActivity(new Intent(requireContext(), AccelerationActivity.class)));
            click("exit", () -> {
                requireContext().startService(new Intent(requireContext(), TranslatorService.class)
                        .setAction(TranslatorService.ACTION_COMMAND).putExtra(TranslatorService.EXTRA_KIND, "EXIT"));
                requireActivity().finishAffinity();
            });
            click("play_samples", this::playSamples);
            tts = new TextToSpeech(requireContext(), st -> requireActivity().runOnUiThread(() -> fillVoices(st == TextToSpeech.SUCCESS)));
        }

        @Override
        public void onDestroy() {
            if (tts != null) {
                tts.shutdown();
            }
            super.onDestroy();
        }

        private void click(String key, Runnable r) {
            Preference p = findPreference(key);
            if (p != null) {
                p.setOnPreferenceClickListener(x -> {
                    r.run();
                    return true;
                });
            }
        }

        /** Phone voices (offline only; male ones first where known) and the downloaded Piper voices. */
        private void fillVoices(boolean ttsOk) {
            ModelStore store = new ModelStore(requireContext());
            for (String lang : new String[] {"it", "en"}) {
                List<String> names = new ArrayList<>();
                List<String> values = new ArrayList<>();
                names.add("Automatic (phone voice if installed, else Piper)");
                values.add("");
                if (ttsOk && tts.getVoices() != null) {
                    List<Voice> vs = new ArrayList<>();
                    for (Voice v : tts.getVoices()) {
                        if (v.getLocale().getLanguage().equals(lang) && !v.isNetworkConnectionRequired()
                                && (v.getFeatures() == null || !v.getFeatures().contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED))) {
                            vs.add(v);
                        }
                    }
                    vs.sort((a, b) -> ("male".equals(SystemTts.gender(b.getName())) ? 1 : 0) - ("male".equals(SystemTts.gender(a.getName())) ? 1 : 0));
                    for (Voice v : vs) {
                        String g = SystemTts.gender(v.getName());
                        names.add("Phone: " + v.getName() + (g.isEmpty() ? "" : " (" + g + ")"));
                        values.add(SystemTts.PREFIX + v.getName());
                    }
                }
                for (Model m : store.catalog().voices(lang)) {
                    if (store.installed(m)) {
                        names.add(m.name() + " (" + m.gender() + ", Piper)");
                        values.add("piper:" + m.id());
                    }
                }
                ListPreference p = findPreference("voice_" + lang);
                if (p != null) {
                    p.setEntries(names.toArray(new String[0]));
                    p.setEntryValues(values.toArray(new String[0]));
                    if (p.getValue() == null) {
                        p.setValue("");
                    }
                }
            }
        }

        private void playSamples() {
            Toast.makeText(requireContext(), "Playing samples…", Toast.LENGTH_SHORT).show();
            android.content.Context ctx = requireContext().getApplicationContext();
            new Thread(() -> {
                AudioTrackPlayer player = new AudioTrackPlayer();
                for (String lang : new String[] {"it", "en"}) {
                    String v = SettingsStore.get(ctx, SettingsStore.VOICE_PREFIX + lang, "");
                    try {
                        AudioClip clip;
                        if (v.startsWith("piper:")) {
                            ModelStore store = new ModelStore(ctx);
                            Model m = store.catalog().get(v.substring(6));
                            try (PiperTts p = new EngineFactory(ctx, store).piper(Map.of(lang, m),
                                    org.myvoice.accel.Choice.cpu(2, ""), 150)) {
                                clip = p.synthesize(SAMPLES.get(lang), lang);
                            }
                        } else {
                            try (SystemTts s = new SystemTts(ctx, Map.of(lang, v))) {
                                clip = s.synthesize(SAMPLES.get(lang), lang);
                            }
                        }
                        player.start(clip);
                        while (player.isPlaying()) {
                            Thread.sleep(50);
                        }
                    } catch (Exception e) {
                        requireActivity().runOnUiThread(() -> Toast.makeText(ctx, lang + ": " + e.getMessage(), Toast.LENGTH_LONG).show());
                    }
                }
            }, "samples").start();
        }
    }
}
