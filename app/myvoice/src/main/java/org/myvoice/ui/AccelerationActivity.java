package org.myvoice.ui;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import android.os.Bundle;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;

import org.myvoice.accel.AccelerationManager;
import org.myvoice.accel.DeviceInfo;
import org.myvoice.accel.ProbeResult;
import org.myvoice.accel.Stage;
import org.myvoice.core.SettingsStore;
import org.myvoice.models.ModelCatalog;
import org.myvoice.models.ModelCatalog.Model;
import org.myvoice.models.ModelStore;

/** What this phone has, and for every model which backend was tried, what happened and which one is used. */
public final class AccelerationActivity extends AppCompatActivity {
    private TextView report;
    private MaterialButton again;

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(32, 32, 32, 32);
        again = new MaterialButton(this);
        again.setText("Test again");
        again.setOnClickListener(v -> rerun());
        report = new TextView(this);
        report.setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodyMedium);
        report.setTextIsSelectable(true);
        root.addView(again);
        root.addView(report);
        ScrollView sv = new ScrollView(this);
        sv.setFitsSystemWindows(true);
        sv.addView(root);
        setContentView(sv);
        show();
    }

    private List<Model> selected(ModelStore store) {
        ModelCatalog cat = store.catalog();
        List<Model> out = new ArrayList<>();
        for (String comp : new String[] {"asr", "mt", "voice_it", "voice_en"}) {
            Model m = cat.get(SettingsStore.get(this, SettingsStore.MODEL_PREFIX + comp, ""));
            if (m != null && store.installed(m)) {
                out.add(m);
            }
        }
        return out;
    }

    private void show() {
        AccelerationManager accel = new AccelerationManager(this);
        DeviceInfo d = accel.deviceInfo();
        StringBuilder s = new StringBuilder();
        for (Map.Entry<String, String> e : d.summary().entrySet()) {
            s.append(e.getKey()).append(": ").append(e.getValue()).append('\n');
        }
        if (!accel.cpuVariant().isEmpty()) {
            s.append("CPU code used by whisper.cpp / llama.cpp: ").append(accel.cpuVariant()).append('\n');
        }
        ModelStore store = new ModelStore(this);
        for (Model m : selected(store)) {
            Stage stage = switch (m.engine()) {
                case "whisper" -> Stage.ASR;
                case "llama" -> Stage.MT_LLM;
                case "marian" -> Stage.MT_ONNX;
                default -> Stage.TTS_PIPER;
            };
            s.append("\n").append(stage.label).append(" · ").append(m.name()).append('\n');
            List<ProbeResult> rs = accel.results(stage, m.id());
            if (rs.isEmpty()) {
                s.append("  not tested yet: runs on the CPU\n");
            }
            for (ProbeResult r : rs) {
                s.append("  ").append(r.line()).append('\n');
            }
        }
        s.append("\nThe phone's own voices are run by its text-to-speech engine.");
        report.setText(s.toString());
    }

    private void rerun() {
        again.setEnabled(false);
        report.setText("Testing… (a few minutes; the conversation keeps its current settings until restarted)\n");
        AccelerationManager accel = new AccelerationManager(this);
        accel.probeAsync(selected(new ModelStore(this)), new AccelerationManager.Listener() {
            @Override
            public void onProgress(String message) {
                runOnUiThread(() -> report.append(message + "\n"));
            }

            @Override
            public void onResult(ProbeResult r) {
            }
        }, err -> runOnUiThread(() -> {
            again.setEnabled(true);
            show();
            if (err != null) {
                report.append("\nTesting failed: " + err);
            }
        }));
    }
}
