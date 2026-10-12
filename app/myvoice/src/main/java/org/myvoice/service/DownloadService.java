package org.myvoice.service;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;

import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.net.ConnectivityManager;
import android.net.NetworkCapabilities;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.PowerManager;

import org.myvoice.core.SettingsStore;
import org.myvoice.models.Downloader;
import org.myvoice.models.ModelCatalog.FileSpec;
import org.myvoice.models.ModelCatalog.Model;
import org.myvoice.models.ModelStore;

/**
 * Downloads the chosen models in a foreground service (type dataSync), so it continues with the
 * screen off. Resumable and checksummed (models/Downloader); the setup screen shows the progress.
 */
public final class DownloadService extends Service {
    public static final String EXTRA_MODELS = "models";

    /** Download progress, for the setup screen (called on the main thread). */
    public interface Listener {
        void onProgress(String file, long fileDone, long fileTotal, long allDone, long allTotal, double bytesPerSecond);

        void onMessage(String text);

        void onFinished(boolean ok, String message);
    }

    private static final List<Listener> LISTENERS = new CopyOnWriteArrayList<>();
    private static final AtomicBoolean CANCEL = new AtomicBoolean();
    private static volatile boolean active;
    private final Handler main = new Handler(Looper.getMainLooper());
    private Notifications notifications;

    public static void addListener(Listener l) {
        LISTENERS.add(l);
    }

    public static void removeListener(Listener l) {
        LISTENERS.remove(l);
    }

    public static boolean isActive() {
        return active;
    }

    public static void cancel() {
        CANCEL.set(true);
    }

    public static void start(Context c, List<String> modelIds) {
        Intent i = new Intent(c, DownloadService.class).putStringArrayListExtra(EXTRA_MODELS, new ArrayList<>(modelIds));
        c.startForegroundService(i);
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        notifications = new Notifications(this);
        startForeground(Notifications.DOWNLOAD_ID, notifications.download("Downloading models", "starting…", -1),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
        if (active) {
            return START_NOT_STICKY;
        }
        List<String> ids = intent != null ? intent.getStringArrayListExtra(EXTRA_MODELS) : null;
        active = true;
        CANCEL.set(false);
        new Thread(() -> run(ids == null ? List.of() : ids), "download").start();
        return START_NOT_STICKY;
    }

    private boolean unmetered() {
        ConnectivityManager cm = getSystemService(ConnectivityManager.class);
        NetworkCapabilities nc = cm.getNetworkCapabilities(cm.getActiveNetwork());
        return nc != null && nc.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED);
    }

    private void post(Runnable r) {
        main.post(r);
    }

    private void run(List<String> ids) {
        PowerManager.WakeLock wl = getSystemService(PowerManager.class)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "myvoice:download");
        wl.acquire(6 * 60 * 60 * 1000L);
        boolean ok = true;
        String message = "All models are here.";
        try {
            if (SettingsStore.prefs(this).getBoolean(SettingsStore.WIFI_ONLY, false) && !unmetered()) {
                throw new Downloader.DownloadException("Wi-Fi only is on and this connection is metered");
            }
            ModelStore store = new ModelStore(this);
            List<FileSpec> files = new ArrayList<>();
            for (String id : ids) {
                Model m = store.catalog().get(id);
                if (m != null) {
                    for (FileSpec f : m.files()) {
                        if (!store.hasFile(f) || !store.isVerified(f)) {
                            files.add(f);
                        }
                    }
                }
            }
            long total = 0;
            for (FileSpec f : files) {
                total += f.size();
            }
            final long allTotal = total;
            long[] doneBefore = {0};
            Downloader dl = new Downloader(CANCEL);
            for (FileSpec f : files) {
                File out = store.file(f);
                if (store.hasFile(f) && !store.isVerified(f)) {
                    // copied over USB: check it once
                    post(() -> message("checking " + f.name() + " (copied over USB)…"));
                    if (!f.sha256().equalsIgnoreCase(Downloader.sha256(out))) {
                        out.delete();
                        post(() -> message(f.name() + " was damaged: downloading it again"));
                    } else {
                        store.markVerified(f);
                        doneBefore[0] += f.size();
                        continue;
                    }
                }
                long[] last = {System.nanoTime(), 0};
                double[] speed = {0};
                long base = doneBefore[0];
                dl.fetch(f.url(), out, f.size(), f.sha256(), new Downloader.Progress() {
                    @Override
                    public void onBytes(long done, long size) {
                        long now = System.nanoTime();
                        double dt = (now - last[0]) / 1e9;
                        if (dt >= 1.0) {
                            double inst = (done - last[1]) / dt;
                            speed[0] = speed[0] == 0 ? inst : 0.7 * speed[0] + 0.3 * inst;
                            last[0] = now;
                            last[1] = done;
                        }
                        long all = base + done;
                        double sp = speed[0];
                        post(() -> {
                            for (Listener l : LISTENERS) {
                                l.onProgress(f.name(), done, size, all, allTotal, sp);
                            }
                        });
                        int pct = allTotal > 0 ? (int) (100 * all / allTotal) : 0;
                        notifications.notify(Notifications.DOWNLOAD_ID, notifications.download("Downloading models",
                                String.format(Locale.ROOT, "%s · %d%%", f.name(), pct), pct));
                    }

                    @Override
                    public void onMessage(String m) {
                        post(() -> message(m));
                    }
                });
                store.markVerified(f);
                doneBefore[0] += f.size();
            }
        } catch (Exception e) {
            ok = false;
            message = CANCEL.get() ? "Download paused; tap Download to continue where it stopped."
                    : (e.getMessage() != null ? e.getMessage() : e.toString());
        } finally {
            wl.release();
        }
        boolean fok = ok;
        String fmsg = message;
        active = false;
        post(() -> {
            for (Listener l : LISTENERS) {
                l.onFinished(fok, fmsg);
            }
            stopForeground(STOP_FOREGROUND_REMOVE);
            stopSelf();
        });
    }

    private void message(String m) {
        for (Listener l : LISTENERS) {
            l.onMessage(m);
        }
    }
}
