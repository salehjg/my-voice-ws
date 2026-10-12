package org.myvoice.service;

import java.io.InputStream;
import java.util.Map;

import android.app.Service;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.Message;
import android.os.Messenger;
import android.os.RemoteException;

import org.myvoice.accel.Backend;
import org.myvoice.accel.Choice;
import org.myvoice.asr.Asr;
import org.myvoice.asr.Transcript;
import org.myvoice.audio.AudioClip;
import org.myvoice.audio.Wav;
import org.myvoice.core.EngineFactory;
import org.myvoice.models.ModelCatalog.Model;
import org.myvoice.models.ModelStore;
import org.myvoice.mt.Translator;
import org.myvoice.tts.PiperTts;

/**
 * Tries one model on one backend, in its own process (android:process=":probe"): a GPU driver that
 * crashes in native code takes down only this process, and the app records "crashed" instead of
 * dying. The same idea as the laptop's check.py running llama.cpp in a subprocess to catch SIGILL.
 */
public final class ProbeService extends Service {
    public static final int MSG_JOB = 1;
    public static final int MSG_RESULT = 2;

    public static final String ASR_TEXT = "Buongiorno, vorrei un caffè e un cornetto, per favore.";
    public static final String MT_TEXT = "Buongiorno, vorrei prenotare un tavolo per due persone stasera alle otto.";
    public static final String TTS_TEXT = "Buongiorno, come sta? Il treno parte alle nove.";

    private HandlerThread thread;
    private Messenger messenger;

    @Override
    public void onCreate() {
        super.onCreate();
        thread = new HandlerThread("probe");
        thread.start();
        messenger = new Messenger(new Handler(thread.getLooper(), this::handle));
    }

    @Override
    public IBinder onBind(Intent intent) {
        return messenger.getBinder();
    }

    @Override
    public void onDestroy() {
        thread.quitSafely();
        super.onDestroy();
    }

    private boolean handle(Message msg) {
        if (msg.what != MSG_JOB) {
            return false;
        }
        Bundle job = msg.getData();
        Bundle out = new Bundle();
        try {
            run(job, out);
            out.putBoolean("ok", true);
        } catch (Throwable t) {
            out.putBoolean("ok", false);
            out.putString("error", t.getClass().getSimpleName() + (t.getMessage() == null ? "" : ": " + t.getMessage()));
        }
        Message reply = Message.obtain(null, MSG_RESULT);
        reply.setData(out);
        try {
            msg.replyTo.send(reply);
        } catch (RemoteException ignored) {
        }
        return true;
    }

    private void run(Bundle job, Bundle out) throws Exception {
        String type = job.getString("type");
        if ("deviceinfo".equals(type)) {
            java.util.List<java.util.List<String>> l = org.myvoice.accel.DeviceInfo.queryAccelerators();
            out.putStringArray("vulkan", l.get(0).toArray(new String[0]));
            out.putStringArray("opencl", l.get(1).toArray(new String[0]));
            out.putStringArray("nnapi", l.get(2).toArray(new String[0]));
            EngineFactory.loadGgml(this, "");   // the best CPU variant for this CPU
            out.putString("cpu", org.myvoice.nativebridge.NativeLib.ggmlCpuFeatures());
            return;
        }
        if ("devices".equals(type)) {
            out.putStringArray("devices", EngineFactory.loadGgml(this, job.getString("gpu", "")));
            return;
        }
        ModelStore store = new ModelStore(this);
        Model model = store.catalog().get(job.getString("model"));
        Backend backend = Backend.valueOf(job.getString("backend"));
        Choice choice = new Choice(backend, job.getString("device", ""), job.getString("device", ""), job.getInt("threads", 4), "");
        EngineFactory f = new EngineFactory(this, store);
        String output;
        double seconds;
        switch (type) {
            case "asr" -> {
                AudioClip clip;
                try (InputStream in = getAssets().open("probe_it.wav")) {
                    clip = Wav.read(in).resampled(16000);
                }
                try (Asr asr = f.asr(model, choice, "", job.getBoolean("shortCtx", true))) {
                    asr.transcribe(clip.samples(), EngineFactory.languages(), "it");   // warm-up: first run allocates
                    long t0 = System.nanoTime();
                    Transcript tr = asr.transcribe(clip.samples(), EngineFactory.languages(), "it");
                    seconds = (System.nanoTime() - t0) / 1e9;
                    output = tr == null ? "" : tr.text();
                }
            }
            case "mt" -> {
                try (Translator mt = f.translator(model, choice)) {
                    mt.translate("Ciao.", "it", "en");
                    long t0 = System.nanoTime();
                    output = mt.translate(MT_TEXT, "it", "en");
                    seconds = (System.nanoTime() - t0) / 1e9;
                }
            }
            case "tts" -> {
                try (PiperTts tts = f.piper(Map.of(model.lang(), model), choice, 150)) {
                    tts.synthesize("Ciao.", model.lang());
                    long t0 = System.nanoTime();
                    AudioClip a = tts.synthesize(TTS_TEXT, model.lang());
                    seconds = (System.nanoTime() - t0) / 1e9;
                    double rms = 0;
                    for (float v : a.samples()) {
                        if (Float.isNaN(v)) {
                            throw new IllegalStateException("the voice produced invalid audio (NaN)");
                        }
                        rms += v * v;
                    }
                    rms = Math.sqrt(rms / Math.max(1, a.samples().length));
                    // a fingerprint the main process compares with the CPU's: length and loudness
                    output = String.format(java.util.Locale.ROOT, "%.1fs rms %.3f", a.seconds(), rms);
                }
            }
            default -> throw new IllegalArgumentException("unknown probe " + type);
        }
        out.putString("output", output);
        out.putDouble("seconds", seconds);
    }
}
