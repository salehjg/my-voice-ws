package org.myvoice.accel;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.SharedPreferences;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.Message;
import android.os.Messenger;
import android.os.RemoteException;

import org.myvoice.BuildConfig;
import org.myvoice.core.MiniJson;
import org.myvoice.models.ModelCatalog.Model;
import org.myvoice.models.ModelStore;
import org.myvoice.service.ProbeService;

/**
 * Finds out, once per phone, which backend each model runs best on, and remembers it.
 *
 * <p>Every candidate (CPU with a few thread counts, Vulkan, OpenCL, NNAPI, QNN, ...) is tried in
 * the separate probe process on a fixed input. A backend is used only if it works, gives the same
 * answer as the CPU, and is the fastest; everything else is reported with the reason.
 */
public final class AccelerationManager {
    private static final String PREFS = "acceleration";
    private static final long JOB_TIMEOUT_S = 600;
    private static final int[] CPU_THREADS = {4, 8};

    private final Context context;
    private final SharedPreferences prefs;
    private final List<ProbeResult> results = new ArrayList<>();
    private List<List<String>> gpuLists;
    private List<String> ggmlDevices = List.of();

    public AccelerationManager(Context c) {
        context = c.getApplicationContext();
        prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        load();
    }

    /** Results are only valid for this phone build and this app version. */
    static String fingerprint() {
        return Build.FINGERPRINT + "|" + BuildConfig.VERSION_CODE;
    }

    private void load() {
        if (!fingerprint().equals(prefs.getString("fingerprint", ""))) {
            return;
        }
        try {
            Object root = new MiniJson(new StringReader(prefs.getString("results", "[]"))).parse();
            for (Object o : (List<?>) root) {
                results.add(ProbeResult.fromJson((Map<?, ?>) o));
            }
            Object gl = new MiniJson(new StringReader(prefs.getString("gpu", "null"))).parse();
            if (gl instanceof List<?> l && l.size() == 3) {
                List<List<String>> lists = new ArrayList<>();
                for (Object x : l) {
                    List<String> one = new ArrayList<>();
                    for (Object y : (List<?>) x) {
                        one.add((String) y);
                    }
                    lists.add(one);
                }
                gpuLists = lists;
            }
            Object dl = new MiniJson(new StringReader(prefs.getString("ggml", "[]"))).parse();
            List<String> d = new ArrayList<>();
            for (Object x : (List<?>) dl) {
                d.add((String) x);
            }
            ggmlDevices = d;
        } catch (IOException | RuntimeException e) {
            results.clear();
        }
    }

    private void save() {
        StringBuilder r = new StringBuilder("[");
        for (int i = 0; i < results.size(); i++) {
            r.append(i > 0 ? "," : "").append(results.get(i).toJson());
        }
        r.append(']');
        prefs.edit().putString("fingerprint", fingerprint()).putString("results", r.toString())
                .putString("gpu", gpuLists == null ? "null" : listJson(gpuLists))
                .putString("ggml", listJson1(ggmlDevices)).apply();
    }

    private static String listJson1(List<String> l) {
        StringBuilder b = new StringBuilder("[");
        for (int i = 0; i < l.size(); i++) {
            b.append(i > 0 ? "," : "").append(org.myvoice.core.Json.quote(l.get(i)));
        }
        return b.append(']').toString();
    }

    private static String listJson(List<List<String>> l) {
        StringBuilder b = new StringBuilder("[");
        for (int i = 0; i < l.size(); i++) {
            b.append(i > 0 ? "," : "").append(listJson1(l.get(i)));
        }
        return b.append(']').toString();
    }

    /** ggml device name ("Vulkan0") -> its description ("Mali-G52 MC2"). */
    public synchronized String label(String device) {
        for (String d : ggmlDevices) {
            String[] f = d.split("\t");
            if (f.length >= 3 && f[1].equals(device)) {
                return f[2];
            }
        }
        return device;
    }

    public synchronized DeviceInfo deviceInfo() {
        return new DeviceInfo(context, gpuLists);
    }

    /** The features of the ggml CPU code the probe loaded, e.g. "NEON DOTPROD FP16_VA". */
    public String cpuVariant() {
        return prefs.getString("cpu_variant", "");
    }

    public synchronized List<ProbeResult> results(Stage stage, String modelId) {
        List<ProbeResult> out = new ArrayList<>();
        for (ProbeResult r : results) {
            if (r.stage() == stage && r.modelId().equals(modelId)) {
                out.add(r);
            }
        }
        return out;
    }

    public synchronized boolean probed(Stage stage, String modelId) {
        return !results(stage, modelId).isEmpty();
    }

    public static int defaultThreads() {
        return Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors()));
    }

    /**
     * The backend to use for a stage. {@code policy} is the user's setting: "auto" (fastest that
     * works), "cpu", or a backend name to force.
     */
    public synchronized Choice choose(Stage stage, String modelId, String policy) {
        List<ProbeResult> rs = results(stage, modelId);
        ProbeResult bestCpu = rs.stream().filter(r -> r.backend() == Backend.CPU && r.usable())
                .min(Comparator.comparingDouble(ProbeResult::seconds)).orElse(null);
        int cpuThreads = bestCpu != null ? bestCpu.threads() : defaultThreads();
        if ("cpu".equals(policy)) {
            return Choice.cpu(cpuThreads, "CPU chosen in Settings");
        }
        if (policy != null && !policy.equals("auto")) {
            Backend forced = Backend.valueOf(policy);
            for (ProbeResult r : rs) {
                if (r.backend() == forced) {
                    if (!r.usable()) {
                        return Choice.cpu(cpuThreads, forced.label + " chosen in Settings, but it " + r.status().label.substring(2)
                                + " here: using the CPU");
                    }
                    return new Choice(forced, r.device(), label(r.device()), r.threads(), "chosen in Settings");
                }
            }
            return Choice.cpu(cpuThreads, forced.label + " chosen in Settings but not tested yet: using the CPU");
        }
        ProbeResult best = rs.stream().filter(ProbeResult::usable).min(Comparator.comparingDouble(ProbeResult::seconds)).orElse(null);
        if (best == null) {
            return Choice.cpu(cpuThreads, rs.isEmpty() ? "not optimised for this phone yet" : "no accelerator worked");
        }
        long tried = rs.stream().filter(r -> r.status() != ProbeResult.Status.UNAVAILABLE).count();
        return new Choice(best.backend(), best.device(), label(best.device()), best.threads(),
                String.format(Locale.ROOT, "fastest of %d tried (%.1f s)", tried, best.seconds()));
    }

    // ------------------------------------------------------------------ probing

    public interface Listener {
        void onProgress(String message);

        void onResult(ProbeResult r);
    }

    /** One probe job: a model on a backend. */
    record Job(Stage stage, Model model, Backend backend, String device, int threads) {
    }

    /**
     * Probe every candidate for the given models (blocking; call off the UI thread). The asr/mt
     * models are tried on CPU (a few thread counts) and on each GPU device ggml found; Opus-MT and
     * Piper on each ONNX Runtime provider.
     */
    public void probe(List<Model> models, Listener l) throws InterruptedException {
        try (ProbeClient client = new ProbeClient(context)) {
            l.onProgress("Checking this phone's accelerators…");
            Bundle info = client.run(job("deviceinfo"));
            synchronized (this) {
                if (info != null && info.getBoolean("ok")) {
                    gpuLists = List.of(List.of(info.getStringArray("vulkan")), List.of(info.getStringArray("opencl")),
                            List.of(info.getStringArray("nnapi")));
                    prefs.edit().putString("cpu_variant", info.getString("cpu", "")).apply();
                } else {
                    String why = info == null ? "the check crashed" : info.getString("error", "failed");
                    List<String> e = List.of("error: " + why);
                    gpuLists = List.of(e, e, e);
                }
            }
            // which GPU backends ggml can actually load (each one alone: a crash then names the culprit)
            List<String> gpuDevices = new ArrayList<>();
            Set<Backend> ggmlGpuOk = new HashSet<>();
            for (String gpu : new String[] {"vulkan", "opencl"}) {
                l.onProgress("Trying " + gpu + "…");
                Bundle b = new Bundle();
                b.putString("type", "devices");
                b.putString("gpu", gpu);
                Bundle out = client.run(b);
                Backend be = gpu.equals("vulkan") ? Backend.VULKAN : Backend.OPENCL;
                if (out == null) {
                    for (Model m : models) {
                        if (m.engine().equals("whisper") || m.engine().equals("llama")) {
                            record(new ProbeResult(m.engine().equals("whisper") ? Stage.ASR : Stage.MT_LLM, m.id(), be, "",
                                    0, ProbeResult.Status.CRASHED, 0, "the " + gpu + " driver crashed while starting"), l);
                        }
                    }
                    continue;
                }
                for (String d : out.getStringArray("devices")) {
                    String[] f = d.split("\t");
                    if (f.length >= 4 && (f[2].equals("gpu") || f[2].equals("igpu")) && f[3].toLowerCase(Locale.ROOT).contains(gpu)) {
                        gpuDevices.add(be.name() + "\t" + f[0] + "\t" + f[1]);
                        ggmlGpuOk.add(be);
                    }
                }
            }
            synchronized (this) {
                ggmlDevices = gpuDevices;
            }
            for (Model m : models) {
                probeModel(client, m, gpuDevices, ggmlGpuOk, l);
            }
        }
        synchronized (this) {
            save();
        }
    }

    private static Bundle job(String type) {
        Bundle b = new Bundle();
        b.putString("type", type);
        return b;
    }

    private synchronized void record(ProbeResult r, Listener l) {
        results.removeIf(x -> x.stage() == r.stage() && x.modelId().equals(r.modelId()) && x.backend() == r.backend()
                && x.threads() == r.threads() && x.device().equals(r.device()));
        results.add(r);
        l.onResult(r);
    }

    private void probeModel(ProbeClient client, Model m, List<String> gpuDevices, Set<Backend> ggmlGpuOk, Listener l)
            throws InterruptedException {
        Stage stage;
        String type;
        switch (m.engine()) {
            case "whisper" -> { stage = Stage.ASR; type = "asr"; }
            case "llama" -> { stage = Stage.MT_LLM; type = "mt"; }
            case "marian" -> { stage = Stage.MT_ONNX; type = "mt"; }
            case "piper" -> { stage = Stage.TTS_PIPER; type = "tts"; }
            default -> { return; }
        }
        synchronized (this) {
            results.removeIf(x -> x.stage() == stage && x.modelId().equals(m.id()));
        }
        List<Job> jobs = new ArrayList<>();
        for (int t : CPU_THREADS) {
            if (t <= Runtime.getRuntime().availableProcessors()) {
                jobs.add(new Job(stage, m, Backend.CPU, "", t));
            }
        }
        for (Backend b : stage.candidates) {
            if (b == Backend.CPU) {
                continue;
            }
            if (b == Backend.VULKAN || b == Backend.OPENCL) {
                if (!ggmlGpuOk.contains(b)) {
                    record(new ProbeResult(stage, m.id(), b, "", 0, ProbeResult.Status.UNAVAILABLE, 0,
                            b == Backend.OPENCL ? "no OpenCL GPU that ggml supports (it is tuned for Adreno)" : "no Vulkan GPU found"), l);
                    continue;
                }
                for (String d : gpuDevices) {
                    String[] f = d.split("\t");
                    if (f[0].equals(b.name())) {
                        jobs.add(new Job(stage, m, b, f[1], defaultThreads()));
                    }
                }
            } else if (!OrtSessions.available(b)) {
                record(new ProbeResult(stage, m.id(), b, "", 0, ProbeResult.Status.UNAVAILABLE, 0,
                        "not in this ONNX Runtime build"), l);
            } else if (b == Backend.NNAPI && !hasNnapiAccelerator()) {
                record(new ProbeResult(stage, m.id(), b, "", 0, ProbeResult.Status.UNAVAILABLE, 0,
                        "no NNAPI GPU/NPU device on this phone"), l);
            } else {
                jobs.add(new Job(stage, m, b, "", defaultThreads()));
            }
        }
        String reference = null;
        List<ProbeResult> done = new ArrayList<>();
        for (Job j : jobs) {
            l.onProgress(m.name() + " on " + j.backend().label + (j.backend() == Backend.CPU ? " ×" + j.threads() : "") + "…");
            Bundle b = job(type);
            b.putString("model", m.id());
            b.putString("backend", j.backend().name());
            b.putString("device", j.device());
            b.putInt("threads", j.threads());
            Bundle out = client.run(b);
            ProbeResult r;
            if (out == null) {
                r = new ProbeResult(stage, m.id(), j.backend(), j.device(), j.threads(), ProbeResult.Status.CRASHED, 0,
                        "the app's probe process crashed (driver problem)");
            } else if (!out.getBoolean("ok")) {
                r = new ProbeResult(stage, m.id(), j.backend(), j.device(), j.threads(), ProbeResult.Status.FAILED, 0,
                        out.getString("error", ""));
            } else {
                String output = out.getString("output", "");
                double secs = out.getDouble("seconds");
                ProbeResult.Status st = ProbeResult.Status.WORKS;
                String detail = output.length() > 60 ? output.substring(0, 60) + "…" : output;
                if (j.backend() == Backend.CPU) {
                    if (reference == null) {
                        reference = output;
                    }
                } else if (reference != null && similarity(reference, output, type) < 0.6) {
                    st = ProbeResult.Status.WRONG;
                    detail = "different from the CPU's answer: " + detail;
                }
                r = new ProbeResult(stage, m.id(), j.backend(), j.device(), j.threads(), st, secs, detail);
            }
            done.add(r);
            record(r, l);
        }
        // mark the winner and the slower ones
        ProbeResult best = done.stream().filter(ProbeResult::usable).min(Comparator.comparingDouble(ProbeResult::seconds)).orElse(null);
        for (ProbeResult r : done) {
            if (!r.usable()) {
                continue;
            }
            record(r.with(r == best ? ProbeResult.Status.USED : ProbeResult.Status.SLOWER), l);
        }
    }

    private boolean hasNnapiAccelerator() {
        if (gpuLists == null) {
            return false;
        }
        for (String d : gpuLists.get(2)) {
            String[] f = d.split("\t");
            if (f.length > 1 && (f[1].equals("GPU") || f[1].equals("NPU/DSP"))) {
                return true;
            }
        }
        return false;
    }

    /** Word overlap for text, length/loudness closeness for voices. */
    static double similarity(String a, String b, String type) {
        if (type.equals("tts")) {
            try {
                double la = Double.parseDouble(a.split("s")[0]), lb = Double.parseDouble(b.split("s")[0]);
                return 1.0 - Math.min(1.0, Math.abs(la - lb) / Math.max(0.1, la));
            } catch (RuntimeException e) {
                return 0;
            }
        }
        Set<String> wa = new HashSet<>(List.of(a.toLowerCase(Locale.ROOT).replaceAll("[^\\p{L}\\p{N} ]", "").split("\\s+")));
        Set<String> wb = new HashSet<>(List.of(b.toLowerCase(Locale.ROOT).replaceAll("[^\\p{L}\\p{N} ]", "").split("\\s+")));
        Set<String> inter = new HashSet<>(wa);
        inter.retainAll(wb);
        Set<String> union = new HashSet<>(wa);
        union.addAll(wb);
        return union.isEmpty() ? 1.0 : inter.size() / (double) union.size();
    }

    // ------------------------------------------------------------------ talking to the probe process

    /** Runs jobs in ProbeService; a crash of that process shows up as a null result. */
    static final class ProbeClient implements AutoCloseable {
        private final Context context;
        private final HandlerThread thread = new HandlerThread("probe-client");
        private Messenger service;
        private ServiceConnection conn;
        private volatile CountDownLatch connected;
        private volatile CountDownLatch replied;
        private volatile Bundle reply;
        private volatile boolean died;
        private final Messenger replyTo;

        ProbeClient(Context c) {
            context = c;
            thread.start();
            replyTo = new Messenger(new Handler(thread.getLooper(), msg -> {
                if (msg.what == ProbeService.MSG_RESULT) {
                    reply = msg.getData();
                    if (replied != null) {
                        replied.countDown();
                    }
                }
                return true;
            }));
        }

        private void bind() throws InterruptedException {
            connected = new CountDownLatch(1);
            died = false;
            conn = new ServiceConnection() {
                @Override
                public void onServiceConnected(ComponentName name, IBinder binder) {
                    service = new Messenger(binder);
                    connected.countDown();
                }

                @Override
                public void onServiceDisconnected(ComponentName name) {
                    died = true;
                    service = null;
                    if (replied != null) {
                        replied.countDown();
                    }
                }
            };
            // BIND_IMPORTANT: the probe process gets the screen's foreground priority. As a plain
            // background process Android keeps it on the slow little cores (cores 0-3 on a Moto G15),
            // which made every CPU timing several times too slow.
            context.bindService(new Intent(context, ProbeService.class), conn, Context.BIND_AUTO_CREATE | Context.BIND_IMPORTANT);
            connected.await(30, TimeUnit.SECONDS);
        }

        private void unbind() {
            if (conn != null) {
                try {
                    context.unbindService(conn);
                } catch (IllegalArgumentException ignored) {
                }
                conn = null;
                service = null;
            }
        }

        /** The job's reply, or null if the probe process crashed or hung. */
        Bundle run(Bundle job) throws InterruptedException {
            if (service == null || died) {
                unbind();
                bind();
            }
            if (service == null) {
                return null;
            }
            reply = null;
            replied = new CountDownLatch(1);
            Message msg = Message.obtain(null, ProbeService.MSG_JOB);
            msg.setData(job);
            msg.replyTo = replyTo;
            try {
                service.send(msg);
            } catch (RemoteException e) {
                died = true;
                return null;
            }
            boolean ok = replied.await(JOB_TIMEOUT_S, TimeUnit.SECONDS);
            if (!ok || died) {
                unbind();   // a hung or crashed probe process: start a fresh one next time
                return null;
            }
            return reply;
        }

        @Override
        public void close() {
            unbind();
            thread.quitSafely();
        }
    }

    /** Small helper for the UI: run probing on a background thread. */
    public Thread probeAsync(List<Model> models, Listener l, Consumer<Throwable> done) {
        Thread t = new Thread(() -> {
            try {
                probe(models, l);
                done.accept(null);
            } catch (Throwable e) {
                done.accept(e);
            }
        }, "probe");
        t.start();
        return t;
    }

    public ModelStore store() {
        return new ModelStore(context);
    }
}
