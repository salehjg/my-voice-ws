package org.myvoice.accel;

import java.util.Locale;
import java.util.Map;

import org.myvoice.core.Json;

/** How one backend did for one model on this phone. */
public record ProbeResult(Stage stage, String modelId, Backend backend, String device, int threads,
                          Status status, double seconds, String detail) {
    public enum Status {
        USED("✓ used"), WORKS("✓ works"), SLOWER("✓ slower"), WRONG("✗ wrong output"), FAILED("✗ failed"),
        CRASHED("✗ crashed"), UNAVAILABLE("– unavailable");

        public final String label;

        Status(String label) {
            this.label = label;
        }
    }

    public boolean usable() {
        return status == Status.USED || status == Status.WORKS || status == Status.SLOWER;
    }

    public ProbeResult with(Status s) {
        return new ProbeResult(stage, modelId, backend, device, threads, s, seconds, detail);
    }

    public String line() {
        String t = seconds > 0 ? String.format(Locale.ROOT, " · %.2f s", seconds) : "";
        String th = backend == Backend.CPU || backend == Backend.XNNPACK ? " ×" + threads : "";
        return status.label + "  " + backend.label + th + (device.isEmpty() ? "" : " (" + device + ")") + t
                + (detail.isEmpty() ? "" : " — " + detail);
    }

    String toJson() {
        return Json.object(Map.of("stage", stage.name(), "model", modelId, "backend", backend.name(), "device", device,
                "threads", threads, "status", status.name(), "seconds", seconds, "detail", detail));
    }

    static ProbeResult fromJson(Map<?, ?> m) {
        return new ProbeResult(Stage.valueOf((String) m.get("stage")), (String) m.get("model"),
                Backend.valueOf((String) m.get("backend")), (String) m.get("device"),
                ((Number) m.get("threads")).intValue(), Status.valueOf((String) m.get("status")),
                ((Number) m.get("seconds")).doubleValue(), (String) m.get("detail"));
    }
}
