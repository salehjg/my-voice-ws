package org.myvoice;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.myvoice.models.Downloader;

/** The port of download.py against a local HTTP server: resume, redirects, drops, bad checksums. */
public class DownloaderTest {
    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    static final byte[] DATA = new byte[300_000];
    static String sha;
    HttpServer server;
    String base;
    final List<String> ranges = new ArrayList<>();
    final AtomicInteger flakyHits = new AtomicInteger();

    static {
        new Random(1).nextBytes(DATA);
        try {
            StringBuilder b = new StringBuilder();
            for (byte x : MessageDigest.getInstance("SHA-256").digest(DATA)) {
                b.append(String.format("%02x", x & 0xff));
            }
            sha = b.toString();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    void serve(HttpExchange ex, boolean honourRange, int dropAfter) throws IOException {
        String range = ex.getRequestHeaders().getFirst("Range");
        synchronized (ranges) {
            ranges.add(range == null ? "" : range);
        }
        int start = 0;
        if (honourRange && range != null) {
            start = Integer.parseInt(range.replace("bytes=", "").replace("-", ""));
            if (start >= DATA.length) {
                ex.sendResponseHeaders(416, -1);
                ex.close();
                return;
            }
            ex.getResponseHeaders().add("Content-Range", "bytes " + start + "-" + (DATA.length - 1) + "/" + DATA.length);
            ex.sendResponseHeaders(206, DATA.length - start);
        } else {
            ex.sendResponseHeaders(200, DATA.length);
        }
        try (OutputStream o = ex.getResponseBody()) {
            int end = dropAfter > 0 ? Math.min(DATA.length, start + dropAfter) : DATA.length;
            o.write(DATA, start, end - start);
            if (end < DATA.length) {
                o.flush();
                throw new IOException("simulated drop");
            }
        }
    }

    @Before
    public void start() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/file", ex -> serve(ex, true, 0));
        server.createContext("/norange", ex -> serve(ex, false, 0));
        server.createContext("/redirect", ex -> {
            ex.getResponseHeaders().add("Location", "/file");
            ex.sendResponseHeaders(302, -1);
            ex.close();
        });
        server.createContext("/flaky", ex -> serve(ex, true, flakyHits.getAndIncrement() == 0 ? 100_000 : 0));
        server.createContext("/missing", ex -> {
            ex.sendResponseHeaders(404, -1);
            ex.close();
        });
        server.start();
        base = "http://127.0.0.1:" + server.getAddress().getPort();
    }

    @After
    public void stop() {
        server.stop(0);
    }

    static final Downloader.Progress QUIET = new Downloader.Progress() {
        @Override public void onBytes(long done, long total) { }
        @Override public void onMessage(String message) { }
    };

    Downloader dl() throws Exception {
        Downloader d = new Downloader(new AtomicBoolean());
        java.lang.reflect.Field f = Downloader.class.getDeclaredField("retryMs");
        f.setAccessible(true);
        f.setLong(d, 50);
        return d;
    }

    @Test
    public void downloadsAndChecks() throws Exception {
        File out = new File(tmp.getRoot(), "a/model.bin");
        dl().fetch(base + "/file", out, DATA.length, sha, QUIET);
        assertArrayEquals(DATA, Files.readAllBytes(out.toPath()));
        assertFalse(new File(out.getPath() + ".part").exists());
    }

    @Test
    public void resumesAPartialFile() throws Exception {
        File out = new File(tmp.getRoot(), "model.bin");
        try (FileOutputStream o = new FileOutputStream(out.getPath() + ".part")) {
            o.write(DATA, 0, 120_000);
        }
        dl().fetch(base + "/file", out, DATA.length, sha, QUIET);
        assertArrayEquals(DATA, Files.readAllBytes(out.toPath()));
        assertEquals(List.of("bytes=120000-"), ranges);
    }

    @Test
    public void startsOverWhenTheServerIgnoresRanges() throws Exception {
        File out = new File(tmp.getRoot(), "model.bin");
        try (FileOutputStream o = new FileOutputStream(out.getPath() + ".part")) {
            o.write(DATA, 0, 120_000);
        }
        dl().fetch(base + "/norange", out, DATA.length, sha, QUIET);
        assertArrayEquals(DATA, Files.readAllBytes(out.toPath()));
    }

    @Test
    public void followsRedirects() throws Exception {
        File out = new File(tmp.getRoot(), "model.bin");
        dl().fetch(base + "/redirect", out, DATA.length, sha, QUIET);
        assertArrayEquals(DATA, Files.readAllBytes(out.toPath()));
    }

    @Test
    public void retriesAfterADroppedConnection() throws Exception {
        File out = new File(tmp.getRoot(), "model.bin");
        List<String> messages = new ArrayList<>();
        dl().fetch(base + "/flaky", out, DATA.length, sha, new Downloader.Progress() {
            @Override public void onBytes(long done, long total) { }
            @Override public void onMessage(String m) { messages.add(m); }
        });
        assertArrayEquals(DATA, Files.readAllBytes(out.toPath()));
        assertTrue(String.join("|", ranges), ranges.size() >= 2 && ranges.get(1).startsWith("bytes="));
        assertTrue(messages.toString(), messages.stream().anyMatch(m -> m.startsWith("connection dropped")));
    }

    @Test
    public void badChecksumDeletesThePartialFile() throws Exception {
        File out = new File(tmp.getRoot(), "model.bin");
        Downloader.DownloadException e = assertThrows(Downloader.DownloadException.class,
                () -> dl().fetch(base + "/file", out, DATA.length, "0".repeat(64), QUIET));
        assertTrue(e.getMessage(), e.getMessage().contains("checksum"));
        assertFalse(out.exists());
        assertFalse(new File(out.getPath() + ".part").exists());
    }

    @Test
    public void missingFileFailsAtOnce() throws Exception {
        File out = new File(tmp.getRoot(), "model.bin");
        Downloader.DownloadException e = assertThrows(Downloader.DownloadException.class,
                () -> dl().fetch(base + "/missing", out, DATA.length, sha, QUIET));
        assertTrue(e.getMessage(), e.getMessage().contains("404"));
    }

    @Test
    public void cancelStops() throws Exception {
        File out = new File(tmp.getRoot(), "model.bin");
        Downloader d = new Downloader(new AtomicBoolean(true));
        assertThrows(Downloader.DownloadException.class, () -> d.fetch(base + "/file", out, DATA.length, sha, QUIET));
    }
}
