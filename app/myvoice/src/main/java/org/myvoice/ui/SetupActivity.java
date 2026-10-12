package org.myvoice.ui;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import android.content.Intent;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.speech.tts.Voice;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.card.MaterialCardView;
import com.google.android.material.progressindicator.LinearProgressIndicator;

import org.myvoice.R;
import org.myvoice.accel.AccelerationManager;
import org.myvoice.accel.DeviceInfo;
import org.myvoice.accel.ProbeResult;
import org.myvoice.core.SettingsStore;
import org.myvoice.models.Estimator;
import org.myvoice.models.ModelCatalog;
import org.myvoice.models.ModelCatalog.Model;
import org.myvoice.models.ModelStore;
import org.myvoice.service.DownloadService;

/**
 * First start (and Settings → Models): pick the models, download them, then let the app find the
 * fastest way to run them on this phone. Defaults to the laptop's models; each choice shows its
 * size, whether it fits in memory and a rough time per sentence on this phone.
 */
public final class SetupActivity extends AppCompatActivity implements DownloadService.Listener {
    private static final String[] COMPONENTS = {"asr", "mt", "voice_it", "voice_en"};
    private static final String[] LABELS = {"Speech recognition", "Translation", "Italian voice (fallback)", "English voice (fallback)"};

    private ModelStore store;
    private ModelCatalog cat;
    private Estimator estimator;
    private DeviceInfo device;
    private final Map<String, Spinner> pickers = new LinkedHashMap<>();
    private final Map<String, List<Model>> options = new LinkedHashMap<>();
    private RadioGroup presets;
    private TextView total, status, fileLine, probeLog, voicesLine;
    private LinearProgressIndicator progress;
    private MaterialButton download, optimise, start;
    private boolean updating;

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        store = new ModelStore(this);
        cat = store.catalog();
        device = new DeviceInfo(this, null);
        setTitle(R.string.setup_title);
        setContentView(build());
        new Thread(() -> {
            Estimator e = Estimator.measure();
            runOnUiThread(() -> {
                estimator = e;
                refresh();
            });
        }, "estimate").start();
        checkSystemVoices();
        refresh();
    }

    @Override
    protected void onStart() {
        super.onStart();
        DownloadService.addListener(this);
    }

    @Override
    protected void onStop() {
        DownloadService.removeListener(this);
        super.onStop();
    }

    // ------------------------------------------------------------------ layout

    private TextView text(String s, int appearance) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextAppearance(appearance);
        t.setPadding(0, 8, 0, 8);
        return t;
    }

    private MaterialCardView card(View... children) {
        MaterialCardView c = new MaterialCardView(this);
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.VERTICAL);
        l.setPadding(32, 24, 32, 24);
        for (View v : children) {
            l.addView(v);
        }
        c.addView(l);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.setMargins(0, 12, 0, 12);
        c.setLayoutParams(lp);
        return c;
    }

    private View build() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(32, 32, 32, 32);

        root.addView(text("Set up myvoice", com.google.android.material.R.style.TextAppearance_Material3_HeadlineSmall));
        StringBuilder d = new StringBuilder();
        for (Map.Entry<String, String> e : device.summary().entrySet()) {
            if (!e.getValue().startsWith("unavailable (not checked")) {
                d.append(e.getKey()).append(": ").append(e.getValue()).append('\n');
            }
        }
        root.addView(card(text("This phone", com.google.android.material.R.style.TextAppearance_Material3_TitleMedium),
                text(d.toString().trim(), com.google.android.material.R.style.TextAppearance_Material3_BodyMedium)));

        presets = new RadioGroup(this);
        for (ModelCatalog.Preset p : cat.presets) {
            RadioButton rb = new RadioButton(this);
            rb.setText(p.name() + " — " + p.note());
            rb.setTag(p.id());
            rb.setId(View.generateViewId());
            presets.addView(rb);
        }
        presets.setOnCheckedChangeListener((g, id) -> {
            if (updating || id == -1) {
                return;
            }
            RadioButton rb = g.findViewById(id);
            applyPreset(cat.preset((String) rb.getTag()));
        });
        LinearLayout pick = new LinearLayout(this);
        pick.setOrientation(LinearLayout.VERTICAL);
        for (int i = 0; i < COMPONENTS.length; i++) {
            String comp = COMPONENTS[i];
            List<Model> ms = comp.startsWith("voice_") ? cat.voices(comp.substring(6)) : cat.component(comp);
            options.put(comp, ms);
            Spinner sp = new Spinner(this);
            sp.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
                @Override
                public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                    // only a real change: setting the selection from refresh() also lands here
                    if (!updating && !ms.get(position).id().equals(currentId(comp))) {
                        SettingsStore.put(SetupActivity.this, SettingsStore.MODEL_PREFIX + comp, ms.get(position).id());
                        refresh();
                    }
                }

                @Override
                public void onNothingSelected(AdapterView<?> parent) {
                }
            });
            pickers.put(comp, sp);
            pick.addView(text(LABELS[i], com.google.android.material.R.style.TextAppearance_Material3_LabelLarge));
            pick.addView(sp);
        }
        total = text("", com.google.android.material.R.style.TextAppearance_Material3_BodyMedium);
        root.addView(card(text("Models", com.google.android.material.R.style.TextAppearance_Material3_TitleMedium), presets, pick, total));

        voicesLine = text("Checking the phone's own voices…", com.google.android.material.R.style.TextAppearance_Material3_BodyMedium);
        MaterialButton installVoices = new MaterialButton(this, null, com.google.android.material.R.attr.materialButtonOutlinedStyle);
        installVoices.setText("Install the phone's voice data");
        installVoices.setOnClickListener(v -> startActivity(new Intent(TextToSpeech.Engine.ACTION_INSTALL_TTS_DATA)));
        root.addView(card(text("Voices", com.google.android.material.R.style.TextAppearance_Material3_TitleMedium), voicesLine, installVoices));

        CheckBox wifi = new CheckBox(this);
        wifi.setText("Download over Wi-Fi only");
        wifi.setChecked(SettingsStore.prefs(this).getBoolean(SettingsStore.WIFI_ONLY, false));
        wifi.setOnCheckedChangeListener((c, on) -> SettingsStore.setBoolean(this, SettingsStore.WIFI_ONLY, on));
        progress = new LinearProgressIndicator(this);
        progress.setMax(1000);
        fileLine = text("", com.google.android.material.R.style.TextAppearance_Material3_BodySmall);
        status = text("", com.google.android.material.R.style.TextAppearance_Material3_BodyMedium);
        download = new MaterialButton(this);
        download.setOnClickListener(v -> onDownload());
        TextView usb = text("Files already on your computer can be copied instead (USB debugging on):\nadb push models/llm/translategemma-4b-it.Q6_K.gguf "
                + store.root.getPath().replace("/storage/emulated/0", "/sdcard") + "/llm/\nthen tap Download / check: copied files are checked, not downloaded.",
                com.google.android.material.R.style.TextAppearance_Material3_BodySmall);
        usb.setTextIsSelectable(true);
        root.addView(card(text("Download", com.google.android.material.R.style.TextAppearance_Material3_TitleMedium), wifi, progress,
                fileLine, status, download, usb));

        probeLog = text("", com.google.android.material.R.style.TextAppearance_Material3_BodySmall);
        optimise = new MaterialButton(this);
        optimise.setText("Optimise for this phone");
        optimise.setOnClickListener(v -> onOptimise());
        start = new MaterialButton(this);
        start.setText("Start");
        start.setOnClickListener(v -> {
            SettingsStore.setBoolean(this, SettingsStore.SETUP_DONE, true);
            startActivity(new Intent(this, MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP));
            finish();
        });
        root.addView(card(text("Speed", com.google.android.material.R.style.TextAppearance_Material3_TitleMedium),
                text("Tries each model on the GPU, NPU and CPU and keeps the fastest that works. Takes a few minutes.",
                        com.google.android.material.R.style.TextAppearance_Material3_BodySmall), optimise, probeLog, start));

        ScrollView sv = new ScrollView(this);
        sv.setFitsSystemWindows(true);
        sv.addView(root);
        return sv;
    }

    // ------------------------------------------------------------------ state

    private List<Model> selected() {
        List<Model> out = new ArrayList<>();
        for (String comp : COMPONENTS) {
            Model m = cat.get(currentId(comp));
            if (m != null) {
                out.add(m);
            }
        }
        return out;
    }

    private String currentId(String comp) {
        String id = SettingsStore.get(this, SettingsStore.MODEL_PREFIX + comp, "");
        if (!id.isEmpty() && cat.get(id) != null) {
            return id;
        }
        for (String m : cat.preset(cat.defaultPreset).models()) {
            Model x = cat.get(m);
            if (x == null) {
                continue;
            }
            if (comp.startsWith("voice_") ? x.component().equals("voice") && x.lang().equals(comp.substring(6)) : x.component().equals(comp)) {
                return m;
            }
        }
        return options.get(comp).get(0).id();
    }

    private void applyPreset(ModelCatalog.Preset p) {
        for (String id : p.models()) {
            Model m = cat.get(id);
            if (m == null) {
                continue;
            }
            String comp = m.component().equals("voice") ? "voice_" + m.lang() : m.component();
            SettingsStore.put(this, SettingsStore.MODEL_PREFIX + comp, id);
        }
        refresh();
    }

    private static org.myvoice.accel.Stage stageOf(Model m) {
        return switch (m.engine()) {
            case "whisper" -> org.myvoice.accel.Stage.ASR;
            case "llama" -> org.myvoice.accel.Stage.MT_LLM;
            case "marian" -> org.myvoice.accel.Stage.MT_ONNX;
            default -> org.myvoice.accel.Stage.TTS_PIPER;
        };
    }

    /** Measured by the probe on this phone if it was, else the estimate; NaN while estimating. */
    private double secondsFor(Model m, AccelerationManager accel) {
        double best = Double.NaN;
        for (ProbeResult r : accel.results(stageOf(m), m.id())) {
            if (r.usable() && (Double.isNaN(best) || r.seconds() < best)) {
                best = r.seconds();
            }
        }
        if (!Double.isNaN(best) || estimator == null) {
            return best;
        }
        boolean reference = android.os.Build.MODEL.toLowerCase(Locale.ROOT).contains(cat.referenceDevice);
        return estimator.seconds(m, cat, reference);
    }

    private String describe(Model m, AccelerationManager accel) {
        StringBuilder s = new StringBuilder(m.name());
        s.append(String.format(Locale.ROOT, " · %.0f MB", m.size() / 1e6));
        double secs = secondsFor(m, accel);
        if (!Double.isNaN(secs) && !m.component().equals("voice")) {
            boolean measured = accel.probed(stageOf(m), m.id());
            s.append(" · ").append(measured ? String.format(Locale.ROOT, "measured %.1f s", secs) : Estimator.describe(secs));
        }
        if (!m.component().equals("voice")) {
            s.append(" · ").append(Estimator.ramFit(device.ramBytes, m.ramMb()));
        }
        s.append(store.installed(m) ? " · ✓ downloaded" : "");
        return s.toString();
    }

    private void refresh() {
        updating = true;
        AccelerationManager accel = new AccelerationManager(this);
        for (String comp : COMPONENTS) {
            List<Model> ms = options.get(comp);
            List<String> labels = new ArrayList<>();
            for (Model m : ms) {
                labels.add(describe(m, accel));
            }
            Spinner sp = pickers.get(comp);
            @SuppressWarnings("unchecked")
            ArrayAdapter<String> a = (ArrayAdapter<String>) sp.getAdapter();
            if (a == null) {
                a = new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, new ArrayList<>(labels));
                sp.setAdapter(a);
            } else {
                a.clear();
                a.addAll(labels);   // just new labels (estimates, "downloaded"); same items
            }
            String id = currentId(comp);
            for (int i = 0; i < ms.size(); i++) {
                if (ms.get(i).id().equals(id) && sp.getSelectedItemPosition() != i) {
                    sp.setSelection(i, false);
                }
            }
        }
        presets.clearCheck();
        for (int i = 0; i < presets.getChildCount(); i++) {
            RadioButton rb = (RadioButton) presets.getChildAt(i);
            ModelCatalog.Preset p = cat.preset((String) rb.getTag());
            boolean all = true;
            for (String id : p.models()) {
                Model m = cat.get(id);
                String comp = m.component().equals("voice") ? "voice_" + m.lang() : m.component();
                all &= currentId(comp).equals(id);
            }
            if (all) {
                presets.check(rb.getId());
            }
        }
        updating = false;

        long missing = 0, size = 0;
        double secs = 0;
        int ram = 0;
        boolean allHere = true;
        for (Model m : selected()) {
            missing += store.missingBytes(m);
            size += m.size();
            ram += m.ramMb();
            allHere &= store.installed(m);
            double one = secondsFor(m, accel);
            if (!Double.isNaN(one) && !m.component().equals("voice")) {
                secs += one;
            }
        }
        total.setText(String.format(Locale.ROOT, "Total %.1f GB · %s to download · %.1f GB free%s",
                size / 1e9, missing == 0 ? "nothing" : String.format(Locale.ROOT, "%.1f GB", missing / 1e9),
                store.freeBytes() / 1e9,
                secs == 0 ? "" : "\nOn this phone: " + Estimator.describe(secs) + " after someone stops talking · "
                        + Estimator.ramFit(device.ramBytes, ram)));
        boolean busy = DownloadService.isActive();
        download.setText(busy ? "Pause download" : allHere ? "Check files" : "Download");
        optimise.setEnabled(allHere && !busy);
        boolean probedAll = allHere;
        for (Model m : selected()) {
            if (m.component().equals("voice")) {
                continue;
            }
            probedAll &= accel.probed(m.engine().equals("whisper") ? org.myvoice.accel.Stage.ASR
                    : m.engine().equals("marian") ? org.myvoice.accel.Stage.MT_ONNX : org.myvoice.accel.Stage.MT_LLM, m.id());
        }
        start.setEnabled(allHere);
        start.setText(probedAll ? "Start" : "Start (not optimised yet: runs on the CPU)");
    }

    private void onDownload() {
        if (DownloadService.isActive()) {
            DownloadService.cancel();
            return;
        }
        List<String> ids = new ArrayList<>();
        for (Model m : selected()) {
            ids.add(m.id());
        }
        status.setText("Starting…");
        DownloadService.start(this, ids);
        download.setText("Pause download");
    }

    private void onOptimise() {
        optimise.setEnabled(false);
        start.setEnabled(false);
        probeLog.setText("");
        AccelerationManager accel = new AccelerationManager(this);
        accel.probeAsync(selected(), new AccelerationManager.Listener() {
            @Override
            public void onProgress(String message) {
                runOnUiThread(() -> status.setText(message));
            }

            @Override
            public void onResult(ProbeResult r) {
                runOnUiThread(() -> probeLog.append(cat.get(r.modelId()).name() + ": " + r.line() + "\n"));
            }
        }, err -> runOnUiThread(() -> {
            status.setText(err == null ? "Done: the fastest working choice is used for each model." : "Optimising failed: " + err);
            refresh();
        }));
    }

    private void checkSystemVoices() {
        TextToSpeech[] holder = new TextToSpeech[1];
        holder[0] = new TextToSpeech(this, st -> {
            if (st != TextToSpeech.SUCCESS) {
                voicesLine.setText("No text-to-speech engine on this phone: the Piper voices below are used.");
                return;
            }
            int it = 0, en = 0;
            if (holder[0].getVoices() != null) {
                for (Voice v : holder[0].getVoices()) {
                    boolean offline = !v.isNetworkConnectionRequired() && (v.getFeatures() == null
                            || !v.getFeatures().contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED));
                    if (offline && v.getLocale().getLanguage().equals("it")) {
                        it++;
                    }
                    if (offline && v.getLocale().getLanguage().equals("en")) {
                        en++;
                    }
                }
            }
            String line = "The phone's own voices (used first): Italian " + (it > 0 ? "✓ " + it : "✗ missing")
                    + " · English " + (en > 0 ? "✓ " + en : "✗ missing")
                    + (it == 0 || en == 0 ? "\nMissing ones use the Piper voice below until you install the voice data." : "");
            voicesLine.setText(line);
            holder[0].shutdown();
        });
    }

    // ------------------------------------------------------------------ DownloadService.Listener

    @Override
    public void onProgress(String file, long fileDone, long fileTotal, long allDone, long allTotal, double bytesPerSecond) {
        progress.setProgressCompat(allTotal > 0 ? (int) (1000 * allDone / allTotal) : 0, true);
        String eta = bytesPerSecond > 0 ? String.format(Locale.ROOT, " · %.1f MB/s · %d min left", bytesPerSecond / 1e6,
                (long) ((allTotal - allDone) / bytesPerSecond / 60)) : "";
        fileLine.setText(String.format(Locale.ROOT, "%s: %.0f / %.0f MB · total %.2f / %.2f GB%s", file, fileDone / 1e6,
                fileTotal / 1e6, allDone / 1e9, allTotal / 1e9, eta));
        download.setText("Pause download");
    }

    @Override
    public void onMessage(String text) {
        status.setText(text);
    }

    @Override
    public void onFinished(boolean ok, String message) {
        status.setText(message);
        refresh();
    }
}
