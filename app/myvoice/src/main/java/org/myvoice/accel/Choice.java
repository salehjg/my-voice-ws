package org.myvoice.accel;

/** The backend picked for a stage. {@code reason} says why (shown in the status). */
public record Choice(Backend backend, String device, String deviceLabel, int threads, String reason) {
    public static Choice cpu(int threads, String reason) {
        return new Choice(Backend.CPU, "", "CPU", threads, reason);
    }

    public String label() {
        return switch (backend) {
            case CPU, XNNPACK -> backend.label + " ×" + threads;
            case SYSTEM -> backend.label;
            default -> deviceLabel.isEmpty() ? backend.label : backend.label + " · " + deviceLabel;
        };
    }
}
