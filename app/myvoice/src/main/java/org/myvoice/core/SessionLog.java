package org.myvoice.core;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/** One JSONL transcript per conversation, like the laptop's outputs/talk-*.jsonl. */
public final class SessionLog {
    private final File file;

    public SessionLog(File dir, String prefix) {
        if (dir == null) {
            file = null;
            return;
        }
        dir.mkdirs();
        file = new File(dir, prefix + "-" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")) + ".jsonl");
    }

    public File file() {
        return file;
    }

    public synchronized void write(TurnResult r) {
        if (file == null) {
            return;
        }
        try (Writer w = new OutputStreamWriter(new FileOutputStream(file, true), StandardCharsets.UTF_8)) {
            w.write(r.toJson());
            w.write('\n');
        } catch (IOException ignored) {
            // a transcript is nice to have; never stop the conversation for it
        }
    }
}
