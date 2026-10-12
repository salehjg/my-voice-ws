package org.myvoice.service;

import java.util.List;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;

import org.myvoice.R;
import org.myvoice.core.Command;
import org.myvoice.core.ConversationListener.State;
import org.myvoice.ui.MainActivity;

/** The ongoing notification: mode, state, last translation, what runs where, and Pause/Exit buttons. */
public final class Notifications {
    public static final int ID = 1;
    public static final int DOWNLOAD_ID = 2;
    public static final String CONVERSATION = "conversation";
    public static final String DOWNLOADS = "downloads";
    private final Context context;
    private final NotificationManager nm;

    public Notifications(Context c) {
        context = c;
        nm = c.getSystemService(NotificationManager.class);
        NotificationChannel conv = new NotificationChannel(CONVERSATION, c.getString(R.string.channel_conversation),
                NotificationManager.IMPORTANCE_LOW);
        conv.setShowBadge(false);
        nm.createNotificationChannel(conv);
        NotificationChannel dl = new NotificationChannel(DOWNLOADS, c.getString(R.string.channel_downloads),
                NotificationManager.IMPORTANCE_LOW);
        nm.createNotificationChannel(dl);
    }

    private PendingIntent command(String kind, int code) {
        Intent i = new Intent(context, TranslatorService.class).setAction(TranslatorService.ACTION_COMMAND)
                .putExtra(TranslatorService.EXTRA_KIND, kind);
        return PendingIntent.getService(context, code, i, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
    }

    private PendingIntent openApp() {
        Intent i = new Intent(context, MainActivity.class).setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        return PendingIntent.getActivity(context, 0, i, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
    }

    static String stateText(State s, String detail) {
        return switch (s) {
            case STARTING -> detail.isEmpty() ? "Starting…" : detail;
            case LISTENING -> "● Listening";
            case HEARING -> "● Hearing…";
            case PAUSED -> "⏸ Paused";
            case TRANSLATING -> "Translating…";
            case SPEAKING -> "🔊 Speaking " + detail;
            case READY -> "Ready: tap Listen to them";
            case RECORDING -> "● Listening to " + detail;
            case CONFIRM -> "Check what I heard";
            case REPLY -> "Your turn";
            case STOPPED -> "Stopped";
        };
    }

    public Notification conversation(String mode, State state, String detail, String lastLine, List<String> chips) {
        Notification.Builder b = new Notification.Builder(context, CONVERSATION)
                .setSmallIcon(R.drawable.ic_stat_myvoice)
                .setContentTitle((mode.equals("auto") ? "Auto" : "Manual") + " · " + stateText(state, detail))
                .setContentText(lastLine.isEmpty() ? String.join(" · ", chips) : lastLine)
                .setStyle(new Notification.BigTextStyle().bigText(
                        (lastLine.isEmpty() ? "" : lastLine + "\n") + String.join("\n", chips)))
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .setContentIntent(openApp())
                .setForegroundServiceBehavior(Notification.FOREGROUND_SERVICE_IMMEDIATE);
        if (mode.equals("auto")) {
            boolean paused = state == State.PAUSED;
            b.addAction(new Notification.Action.Builder(null, paused ? "Resume" : "Pause",
                    command(Command.Kind.TOGGLE_PAUSE.name(), 1)).build());
        } else {
            b.addAction(new Notification.Action.Builder(null, "Listen to them", command(Command.Kind.LISTEN_THEM.name(), 2)).build());
            b.addAction(new Notification.Action.Builder(null, "My turn", command(Command.Kind.MY_TURN.name(), 3)).build());
        }
        b.addAction(new Notification.Action.Builder(null, "Exit", command("EXIT", 4)).build());
        return b.build();
    }

    public void update(Notification n) {
        nm.notify(ID, n);
    }

    public Notification download(String title, String text, int percent) {
        return new Notification.Builder(context, DOWNLOADS)
                .setSmallIcon(R.drawable.ic_stat_myvoice)
                .setContentTitle(title)
                .setContentText(text)
                .setProgress(100, Math.max(0, percent), percent < 0)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .build();
    }

    public void notify(int id, Notification n) {
        nm.notify(id, n);
    }

    public void cancel(int id) {
        nm.cancel(id);
    }
}
