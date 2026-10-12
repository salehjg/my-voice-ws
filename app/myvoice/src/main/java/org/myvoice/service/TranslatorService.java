package org.myvoice.service;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import android.app.Notification;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.media.AudioDeviceCallback;
import android.media.AudioDeviceInfo;
import android.media.AudioManager;
import android.os.Binder;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.PowerManager;

import org.myvoice.accel.AccelerationManager;
import org.myvoice.accel.Choice;
import org.myvoice.accel.Stage;
import org.myvoice.audio.AudioRecordMic;
import org.myvoice.audio.AudioTrackPlayer;
import org.myvoice.audio.FileMic;
import org.myvoice.audio.MicSource;
import org.myvoice.core.Command;
import org.myvoice.core.CommandQueue;
import org.myvoice.core.Conversation;
import org.myvoice.core.ConversationListener;
import org.myvoice.core.Engine;
import org.myvoice.core.EngineFactory;
import org.myvoice.core.SessionLog;
import org.myvoice.core.Settings;
import org.myvoice.core.SettingsStore;
import org.myvoice.core.TurnResult;
import org.myvoice.models.ModelStore;
import org.myvoice.vad.SileroVad;

/**
 * Runs the conversation as a foreground service (type "microphone"), so it keeps listening and
 * translating with the app in the background or the screen off. The notification shows what is
 * happening and has Pause/Resume and Exit; Exit stops everything and frees the models.
 */
public final class TranslatorService extends Service implements ConversationListener {
    public static final String ACTION_START = "org.myvoice.START";
    public static final String ACTION_COMMAND = "org.myvoice.COMMAND";
    public static final String EXTRA_MODE = "mode";
    public static final String EXTRA_KIND = "kind";

    /** What the screen shows; called on the main thread. */
    public interface Ui extends ConversationListener {
        void onStatus(List<String> chips);

        void onExit();
    }

    public final class LocalBinder extends Binder {
        public TranslatorService service() {
            return TranslatorService.this;
        }
    }

    private static volatile boolean running;
    private final LocalBinder binder = new LocalBinder();
    private final CommandQueue commands = new CommandQueue();
    private final Handler main = new Handler(Looper.getMainLooper());
    private final List<Ui> uis = new CopyOnWriteArrayList<>();
    private final List<TurnResult> history = Collections.synchronizedList(new ArrayList<>());
    private final List<String> chips = new CopyOnWriteArrayList<>();
    private Thread worker;
    private PowerManager.WakeLock wakeLock;
    private Notifications notifications;
    private Settings settings;
    private AudioManager audio;
    private AudioDeviceCallback deviceCallback;
    private AudioManager.OnModeChangedListener modeListener;
    private volatile String mode = "auto";
    private volatile State state = State.STARTING;
    private volatile String stateDetail = "";
    private volatile String lastLine = "";
    private volatile boolean exiting;

    public static boolean isRunning() {
        return running;
    }

    @Override
    public IBinder onBind(android.content.Intent intent) {
        return binder;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        notifications = new Notifications(this);
        audio = getSystemService(AudioManager.class);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String action = intent != null ? intent.getAction() : null;
        if (ACTION_COMMAND.equals(action)) {
            String kind = intent.getStringExtra(EXTRA_KIND);
            if (kind != null) {
                if (kind.equals("EXIT")) {
                    exit();
                } else {
                    commands.put(Command.of(Command.Kind.valueOf(kind)));
                }
            }
            return START_NOT_STICKY;
        }
        if (worker == null) {
            String start = intent != null && intent.getStringExtra(EXTRA_MODE) != null
                    ? intent.getStringExtra(EXTRA_MODE) : SettingsStore.get(this, SettingsStore.START_MODE, "auto");
            try {
                startForeground(Notifications.ID, notification(), ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE);
            } catch (RuntimeException e) {   // microphone permission missing, or started from the background
                stopSelf();
                return START_NOT_STICKY;
            }
            running = true;
            startConversation(start);
        }
        return START_NOT_STICKY;
    }

    private Notification notification() {
        return notifications.conversation(mode, state, stateDetail, lastLine, chips);
    }

    private void updateNotification() {
        if (running && !exiting) {
            notifications.update(notification());
        }
    }

    // ------------------------------------------------------------------ the conversation thread

    private boolean headphonesConnected() {
        for (AudioDeviceInfo d : audio.getDevices(AudioManager.GET_DEVICES_OUTPUTS)) {
            switch (d.getType()) {
                case AudioDeviceInfo.TYPE_WIRED_HEADPHONES, AudioDeviceInfo.TYPE_WIRED_HEADSET, AudioDeviceInfo.TYPE_USB_HEADSET,
                     AudioDeviceInfo.TYPE_BLUETOOTH_A2DP, AudioDeviceInfo.TYPE_BLE_HEADSET:
                    return true;
                default:
                    break;
            }
        }
        return false;
    }

    private void startConversation(String startMode) {
        mode = startMode;
        settings = SettingsStore.load(this, headphonesConnected());
        PowerManager pm = getSystemService(PowerManager.class);
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "myvoice:conversation");
        wakeLock.acquire();
        watchAudio();
        worker = new Thread(this::work, "conversation");
        worker.start();
    }

    private void work() {
        AccelerationManager accel = new AccelerationManager(this);
        EngineLoader loader = new EngineLoader(this, accel);
        Engine engine = null;
        SileroVad vad = null;
        MicSource mic = null;
        try {
            onState(State.STARTING, "loading the models…");
            engine = loader.load(settings, msg -> onState(State.STARTING, msg), this::onSwitch);
            setChips(loader);
            vad = new EngineFactory(this, new ModelStore(this)).vad();
            mic = micSource();
            SessionLog log = new SessionLog(settings.sessionLog ? getExternalFilesDir("sessions") : null, "talk");
            Conversation conv = new Conversation(engine, settings, commands, this, new AudioTrackPlayer(), mic, vad, log);
            conv.run(mode);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (Throwable t) {
            onNote(Note.ERROR, "could not start: " + (t.getMessage() != null ? t.getMessage() : t.toString()));
            onState(State.STOPPED, "");
        } finally {
            if (mic != null) {
                mic.close();
            }
            if (vad != null) {
                vad.close();
            }
            if (engine != null) {
                try {
                    engine.close();
                } catch (Exception ignored) {
                }
            }
            main.post(this::finishService);
        }
    }

    private MicSource micSource() {
        if (SettingsStore.prefs(this).getBoolean(SettingsStore.TEST_AUDIO, false)) {
            File dir = getExternalFilesDir("test_audio");
            File[] wavs = dir != null ? dir.listFiles((d, n) -> n.endsWith(".wav")) : null;
            if (wavs != null && wavs.length > 0) {
                Arrays.sort(wavs);
                onNote(Note.INFO, "test audio input: " + wavs.length + " file(s) from " + dir);
                return new FileMic(Arrays.asList(wavs));
            }
            onNote(Note.WARNING, "test audio input is on, but " + dir + " has no .wav files: using the microphone");
        }
        // the speech-recognition source gives the cleanest signal; with speakers the mic is paused while we
        // speak, and with headphones there is no echo, so the platform echo canceller isn't needed
        return new AudioRecordMic(false);
    }

    private void setChips(EngineLoader loader) {
        chips.clear();
        for (Stage s : new Stage[] {Stage.ASR, Stage.MT_LLM, Stage.MT_ONNX, Stage.TTS_SYSTEM}) {
            String v = loader.status.get(s);
            if (v != null) {
                chips.add(v);
            }
        }
        List<String> copy = new ArrayList<>(chips);
        main.post(() -> {
            for (Ui u : uis) {
                u.onStatus(copy);
            }
            updateNotification();
        });
    }

    private void onSwitch(Stage stage, Choice from, Choice to, String reason) {
        onNote(Note.WARNING, stage.label + ": " + from.label() + " failed (" + reason + ") → now on " + to.label());
        // the chips are refreshed from the engines' own descriptions on the next turn
        main.post(() -> {
            for (Ui u : uis) {
                u.onStatus(new ArrayList<>(chips));
            }
        });
    }

    // ------------------------------------------------------------------ phone events

    private void watchAudio() {
        deviceCallback = new AudioDeviceCallback() {
            @Override
            public void onAudioDevicesAdded(AudioDeviceInfo[] added) {
                refreshDuplex();
            }

            @Override
            public void onAudioDevicesRemoved(AudioDeviceInfo[] removed) {
                refreshDuplex();
            }
        };
        audio.registerAudioDeviceCallback(deviceCallback, main);
        modeListener = m -> {
            if (m == AudioManager.MODE_IN_CALL || m == AudioManager.MODE_IN_COMMUNICATION || m == AudioManager.MODE_RINGTONE) {
                commands.put(Command.of(Command.Kind.PAUSE));
                onNote(Note.INFO, "paused for a phone call");
            }
        };
        audio.addOnModeChangedListener(getMainExecutor(), modeListener);
    }

    private void refreshDuplex() {
        if (settings == null || !"auto".equals(SettingsStore.get(this, SettingsStore.LISTENING, "auto"))) {
            return;
        }
        boolean half = !headphonesConnected();
        if (half != settings.halfDuplex) {
            settings.halfDuplex = half;   // the conversation reads it before each translation
            onNote(Note.INFO, half ? "speaker: the mic pauses while I speak" : "headphones: I keep listening while I speak");
        }
    }

    // ------------------------------------------------------------------ API for the screen

    public CommandQueue commands() {
        return commands;
    }

    public void addUi(Ui ui) {
        uis.add(ui);
        ui.onMode(mode);
        ui.onState(state, stateDetail);
        ui.onStatus(new ArrayList<>(chips));
    }

    public void removeUi(Ui ui) {
        uis.remove(ui);
    }

    public List<TurnResult> history() {
        synchronized (history) {
            return new ArrayList<>(history);
        }
    }

    public String mode() {
        return mode;
    }

    public State state() {
        return state;
    }

    /** Stop listening and speaking, free the models, remove the notification, stop the service. */
    public void exit() {
        if (exiting) {
            return;
        }
        exiting = true;
        commands.put(Command.of(Command.Kind.EXIT));
        if (worker == null) {
            finishService();
        } else {
            // a model that is still loading can't be interrupted; give it a moment, then stop anyway
            main.postDelayed(() -> {
                if (running) {
                    finishService();
                }
                if (worker != null && worker.isAlive()) {
                    // still busy inside a model (a long translation): Exit means exit
                    android.os.Process.killProcess(android.os.Process.myPid());
                }
            }, 15_000);
        }
    }

    private boolean finished;

    private void finishService() {
        if (finished) {
            return;
        }
        finished = true;
        running = false;
        if (wakeLock != null && wakeLock.isHeld()) {
            wakeLock.release();
        }
        if (deviceCallback != null) {
            audio.unregisterAudioDeviceCallback(deviceCallback);
        }
        if (modeListener != null) {
            audio.removeOnModeChangedListener(modeListener);
        }
        for (Ui u : uis) {
            u.onExit();
        }
        stopForeground(STOP_FOREGROUND_REMOVE);
        stopSelf();
        if (exiting) {
            // Exit means exit: once the screen has closed, end the app's processes too (this one and
            // the probe process) instead of leaving them cached by Android.
            main.postDelayed(() -> {
                android.app.ActivityManager am = getSystemService(android.app.ActivityManager.class);
                for (android.app.ActivityManager.RunningAppProcessInfo p : am.getRunningAppProcesses()) {
                    if (p.pid != android.os.Process.myPid() && p.processName.startsWith(getPackageName())) {
                        android.os.Process.killProcess(p.pid);
                    }
                }
                android.os.Process.killProcess(android.os.Process.myPid());
            }, 800);
        }
    }

    @Override
    public void onDestroy() {
        running = false;
        super.onDestroy();
    }

    // ------------------------------------------------------------------ ConversationListener (conversation thread)

    @Override
    public void onMode(String m) {
        mode = m;
        main.post(() -> {
            for (Ui u : uis) {
                u.onMode(m);
            }
            updateNotification();
        });
    }

    @Override
    public void onState(State s, String detail) {
        state = s;
        stateDetail = detail == null ? "" : detail;
        main.post(() -> {
            for (Ui u : uis) {
                u.onState(s, stateDetail);
            }
            updateNotification();
        });
    }

    @Override
    public void onLevel(double db, boolean speaking, double seconds) {
        main.post(() -> {
            for (Ui u : uis) {
                u.onLevel(db, speaking, seconds);
            }
        });
    }

    @Override
    public void onTurn(TurnResult r) {
        history.add(r);
        lastLine = r.tgtText;
        main.post(() -> {
            for (Ui u : uis) {
                u.onTurn(r);
            }
            updateNotification();
        });
    }

    @Override
    public void onNote(Note kind, String text) {
        main.post(() -> {
            for (Ui u : uis) {
                u.onNote(kind, text);
            }
        });
    }

    @Override
    public void onConfirm(String heard) {
        main.post(() -> {
            for (Ui u : uis) {
                u.onConfirm(heard);
            }
        });
    }

    @Override
    public void onSettingChanged(String key, boolean value) {
        SettingsStore.setBoolean(this, key, value);
        main.post(() -> {
            for (Ui u : uis) {
                u.onSettingChanged(key, value);
            }
        });
    }
}
