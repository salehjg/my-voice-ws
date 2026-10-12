package org.myvoice.ui;

import java.util.List;
import java.util.Locale;

import android.Manifest;
import android.content.ComponentName;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.IBinder;
import android.provider.OpenableColumns;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;

import androidx.activity.OnBackPressedCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.google.android.material.bottomsheet.BottomSheetDialog;
import com.google.android.material.chip.Chip;
import com.google.android.material.dialog.MaterialAlertDialogBuilder;

import org.myvoice.R;
import org.myvoice.audio.AudioClip;
import org.myvoice.audio.AudioDecoder;
import org.myvoice.core.Command;
import org.myvoice.core.Command.Kind;
import org.myvoice.core.SettingsStore;
import org.myvoice.core.TurnResult;
import org.myvoice.databinding.ActivityMainBinding;
import org.myvoice.databinding.SheetConfirmBinding;
import org.myvoice.lang.Languages;
import org.myvoice.service.TranslatorService;

/**
 * The conversation screen. Everything happens in TranslatorService; this screen shows it and sends
 * the taps. Leaving the screen keeps the service translating; Exit stops it.
 */
public final class MainActivity extends AppCompatActivity implements TranslatorService.Ui {
    private ActivityMainBinding b;
    private TurnAdapter adapter;
    private TranslatorService service;
    private String mode = "auto";
    private State state = State.STARTING;
    private BottomSheetDialog confirmSheet;
    private boolean bound;

    private final ServiceConnection conn = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder binder) {
            service = ((TranslatorService.LocalBinder) binder).service();
            adapter.set(service.history());
            scrollToEnd();
            service.addUi(MainActivity.this);
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            service = null;
        }
    };

    private final ActivityResultLauncher<String[]> permissions = registerForActivityResult(
            new ActivityResultContracts.RequestMultiplePermissions(), granted -> {
                if (Boolean.TRUE.equals(granted.get(Manifest.permission.RECORD_AUDIO))) {
                    startAndBind();
                } else {
                    b.state.setText(R.string.mic_needed);
                }
            });

    private final ActivityResultLauncher<String> pickAudio = registerForActivityResult(
            new ActivityResultContracts.GetContent(), this::translateFile);

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        if (!SettingsStore.prefs(this).getBoolean(SettingsStore.SETUP_DONE, false)) {
            startActivity(new Intent(this, SetupActivity.class));
            finish();
            return;
        }
        b = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(b.getRoot());
        b.toolbar.inflateMenu(R.menu.main);
        b.toolbar.setOnMenuItemClickListener(item -> {
            if (item.getItemId() == R.id.action_exit) {
                exitApp();
            } else if (item.getItemId() == R.id.action_settings) {
                startActivity(new Intent(this, SettingsActivity.class));
            } else if (item.getItemId() == R.id.action_acceleration) {
                startActivity(new Intent(this, AccelerationActivity.class));
            }
            return true;
        });
        adapter = new TurnAdapter(SettingsStore.load(this, false).theirLanguage);
        b.turns.setLayoutManager(new LinearLayoutManager(this));
        b.turns.setAdapter(adapter);
        b.level.setMax(100);

        mode = SettingsStore.get(this, SettingsStore.START_MODE, "auto");
        b.modeGroup.check(mode.equals("auto") ? R.id.modeAuto : R.id.modeManual);
        b.modeGroup.addOnButtonCheckedListener((g, id, checked) -> {
            if (checked) {
                String m = id == R.id.modeAuto ? "auto" : "manual";
                if (!m.equals(mode)) {
                    send(Command.mode(m));
                }
            }
        });
        b.mainButton.setOnClickListener(v -> onMain());
        b.secondButton.setOnClickListener(v -> onSecond());
        b.typeButton.setOnClickListener(v -> typeDialog());
        b.replayButton.setOnClickListener(v -> send(Command.of(Kind.REPLAY)));
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) {
            @Override
            public void handleOnBackPressed() {
                if (TranslatorService.isRunning()) {
                    Toast.makeText(MainActivity.this, R.string.still_running, Toast.LENGTH_SHORT).show();
                }
                moveTaskToBack(true);
            }
        });
        render();
        ensurePermissionsAndStart();
    }

    private void ensurePermissionsAndStart() {
        boolean mic = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED;
        boolean notif = ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED;
        if (mic && notif) {
            startAndBind();
        } else {
            permissions.launch(new String[] {Manifest.permission.RECORD_AUDIO, Manifest.permission.POST_NOTIFICATIONS});
        }
    }

    /** The service must be started while this screen is visible (Android's rule for the microphone). */
    private void startAndBind() {
        Intent i = new Intent(this, TranslatorService.class).setAction(TranslatorService.ACTION_START)
                .putExtra(TranslatorService.EXTRA_MODE, mode);
        if (!TranslatorService.isRunning()) {
            startForegroundService(i);
        }
        bound = bindService(new Intent(this, TranslatorService.class), conn, BIND_AUTO_CREATE);
    }

    @Override
    protected void onDestroy() {
        if (service != null) {
            service.removeUi(this);
        }
        if (bound) {
            unbindService(conn);
            bound = false;
        }
        if (confirmSheet != null) {
            confirmSheet.dismiss();
        }
        super.onDestroy();
    }

    private void exitApp() {
        if (service != null) {
            service.exit();
        } else {
            stopService(new Intent(this, TranslatorService.class));
            finishAndRemoveTask();
        }
    }

    private void send(Command c) {
        if (service != null) {
            service.commands().put(c);
        }
    }

    private void scrollToEnd() {
        if (adapter.getItemCount() > 0) {
            b.turns.scrollToPosition(adapter.getItemCount() - 1);
        }
    }

    // ------------------------------------------------------------------ buttons depend on mode + state

    private void onMain() {
        if (state == State.SPEAKING) {
            send(Command.of(Kind.STOP_SPEAKING));
        } else if (mode.equals("auto")) {
            send(Command.of(Kind.TOGGLE_PAUSE));
        } else if (state == State.RECORDING) {
            send(Command.of(Kind.DONE));
        } else if (state == State.REPLY) {
            send(Command.of(Kind.MY_TURN));
        } else {
            send(Command.of(Kind.LISTEN_THEM));
        }
    }

    private void onSecond() {
        if (mode.equals("auto")) {
            send(Command.of(Kind.WRONG_LANGUAGE));
        } else if (state == State.RECORDING) {
            send(Command.of(Kind.CANCEL));
        } else if (state == State.REPLY) {
            send(Command.of(Kind.LISTEN_THEM));
        } else if (state == State.READY) {
            send(Command.of(Kind.MY_TURN));
        } else {
            pickAudio.launch("audio/*");
        }
    }

    private void render() {
        String their = Languages.name("it"), mine = Languages.name("en");
        b.modeGroup.check(mode.equals("auto") ? R.id.modeAuto : R.id.modeManual);
        boolean busy = state == State.STARTING || state == State.TRANSLATING;
        b.mainButton.setEnabled(!busy || state == State.SPEAKING);
        if (state == State.SPEAKING) {
            b.mainButton.setText(R.string.stop);
        } else if (mode.equals("auto")) {
            b.mainButton.setText(state == State.PAUSED ? "Resume listening" : "Pause");
            b.secondButton.setText(R.string.wrong_language);
        } else {
            switch (state) {
                case RECORDING -> {
                    b.mainButton.setText("Done");
                    b.secondButton.setText("Cancel");
                }
                case REPLY -> {
                    b.mainButton.setText("My turn (speak " + mine + ")");
                    b.secondButton.setText("Listen again");
                }
                case READY -> {
                    b.mainButton.setText("Listen to them (" + their + ")");
                    b.secondButton.setText("My turn");
                }
                default -> {
                    b.mainButton.setText("Listen to them (" + their + ")");
                    b.secondButton.setText(R.string.voice_file);
                }
            }
        }
        b.level.setVisibility(state == State.LISTENING || state == State.HEARING || state == State.RECORDING
                ? View.VISIBLE : View.INVISIBLE);
    }

    // ------------------------------------------------------------------ dialogs

    private void typeDialog() {
        EditText input = new EditText(this);
        input.setHint("Italian or English");
        input.setPadding(48, 24, 48, 24);
        String[] langs = mode.equals("auto") ? new String[] {"Detect", "Italian", "English"} : new String[] {"English", "Italian"};
        int[] which = {0};
        new MaterialAlertDialogBuilder(this)
                .setTitle("Type a sentence")
                .setView(input)
                .setSingleChoiceItems(langs, 0, (d, w) -> which[0] = w)
                .setPositiveButton("Translate", (d, w) -> {
                    String l = langs[which[0]];
                    String code = l.equals("Italian") ? "it" : l.equals("English") ? "en" : null;
                    send(Command.type(input.getText().toString(), code));
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    private void translateFile(Uri uri) {
        if (uri == null) {
            return;
        }
        String name = "voice message";
        try (Cursor c = getContentResolver().query(uri, new String[] {OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (c != null && c.moveToFirst()) {
                name = c.getString(0);
            }
        } catch (RuntimeException ignored) {
        }
        String fname = name;
        b.state.setText("Reading " + fname + "…");
        new Thread(() -> {
            try {
                AudioClip clip = AudioDecoder.decode16k(this, uri);
                send(Command.file(fname, clip));
            } catch (Exception e) {
                runOnUiThread(() -> showNote(Note.ERROR, "cannot read " + fname + ": " + e.getMessage()));
            }
        }, "decode").start();
    }

    // ------------------------------------------------------------------ TranslatorService.Ui (main thread)

    @Override
    public void onMode(String m) {
        mode = m;
        render();
    }

    @Override
    public void onState(State s, String detail) {
        state = s;
        String text = switch (s) {
            case STARTING -> detail.isEmpty() ? "Starting…" : detail;
            case LISTENING -> "● Listening · " + detail;
            case HEARING -> "● Hearing…";
            case PAUSED -> "⏸ Paused";
            case TRANSLATING -> "Translating…" + (detail.isEmpty() ? "" : " (" + detail + ")");
            case SPEAKING -> "🔊 Speaking " + detail + " — tap Stop to cut it short";
            case READY -> "Ready — tap Listen to them";
            case RECORDING -> "● Listening to " + detail + " — Done when finished";
            case CONFIRM -> "Check what I heard";
            case REPLY -> "Your turn — answer in English";
            case STOPPED -> "Stopped";
        };
        b.state.setText(text);
        if (s != State.CONFIRM && confirmSheet != null && confirmSheet.isShowing()) {
            confirmSheet.dismiss();
        }
        render();
    }

    @Override
    public void onLevel(double db, boolean speaking, double seconds) {
        int v = (int) Math.max(0, Math.min(100, (db + 60) / 50 * 100));
        b.level.setProgressCompat(v, true);
        b.level.setIndicatorColor(ContextCompat.getColor(this, speaking ? R.color.mine : R.color.their));
    }

    @Override
    public void onTurn(TurnResult r) {
        adapter.add(r);
        scrollToEnd();
    }

    @Override
    public void onNote(Note kind, String text) {
        showNote(kind, text);
    }

    private void showNote(Note kind, String text) {
        b.note.setVisibility(View.VISIBLE);
        b.note.setText(text);
        b.note.setTextColor(ContextCompat.getColor(this, kind == Note.ERROR ? R.color.error
                : kind == Note.WARNING ? R.color.warn : R.color.their));
    }

    @Override
    public void onConfirm(String heard) {
        SheetConfirmBinding s = SheetConfirmBinding.inflate(LayoutInflater.from(this));
        s.heard.setText(heard);
        confirmSheet = new BottomSheetDialog(this);
        confirmSheet.setContentView(s.getRoot());
        confirmSheet.setCancelable(false);
        s.ok.setOnClickListener(v -> {
            String t = s.heard.getText() == null ? "" : s.heard.getText().toString().trim();
            send(t.equals(heard) ? Command.of(Kind.CONFIRM_OK) : Command.edit(t));
            confirmSheet.dismiss();
        });
        s.again.setOnClickListener(v -> {
            send(Command.of(Kind.AGAIN));
            confirmSheet.dismiss();
        });
        s.cancel.setOnClickListener(v -> {
            send(Command.of(Kind.CANCEL));
            confirmSheet.dismiss();
        });
        confirmSheet.show();
    }

    @Override
    public void onSettingChanged(String key, boolean value) {
        showNote(Note.INFO, String.format(Locale.ROOT, "%s: %s", key.replace('_', ' '), value ? "on" : "off"));
    }

    @Override
    public void onStatus(List<String> chips) {
        b.chips.removeAllViews();
        for (String c : chips) {
            Chip chip = new Chip(this);
            chip.setText(c);
            chip.setOnClickListener(v -> startActivity(new Intent(this, AccelerationActivity.class)));
            b.chips.addView(chip);
        }
    }

    @Override
    public void onExit() {
        finishAndRemoveTask();
    }
}
