package org.myvoice.accel;

import java.util.EnumSet;
import java.util.Map;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.NNAPIFlags;

/** ONNX Runtime sessions on a chosen backend (QNN, NNAPI, XNNPACK or plain CPU). */
public final class OrtSessions {
    private OrtSessions() {
    }

    public static OrtEnvironment env() {
        return OrtEnvironment.getEnvironment();
    }

    /** Whether this ONNX Runtime build can use {@code b} at all. */
    public static boolean available(Backend b) {
        EnumSet<OrtProvider> ps = OrtEnvironment.getAvailableProviders();
        return switch (b) {
            case CPU -> true;
            case XNNPACK -> ps.contains(OrtProvider.XNNPACK);
            case NNAPI -> ps.contains(OrtProvider.NNAPI);
            case QNN -> ps.contains(OrtProvider.QNN);
            default -> false;
        };
    }

    public static OrtSession.SessionOptions options(Backend b, int threads) throws OrtException {
        OrtSession.SessionOptions o = new OrtSession.SessionOptions();
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        switch (b) {
            case CPU -> o.setIntraOpNumThreads(threads);
            case XNNPACK -> {
                // XNNPACK has its own thread pool; keep ORT's small so they don't fight over cores
                o.addXnnpack(Map.of("intra_op_num_threads", Integer.toString(threads)));
                o.setIntraOpNumThreads(1);
                o.addConfigEntry("session.intra_op.allow_spinning", "0");
            }
            case NNAPI -> {
                // CPU_DISABLED: never use NNAPI's slow reference CPU device; what the accelerator
                // can't run stays on ONNX Runtime's own CPU code.
                o.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16, NNAPIFlags.CPU_DISABLED));
                o.setIntraOpNumThreads(threads);
            }
            case QNN -> {
                o.addQnn(Map.of("backend_path", "libQnnHtp.so", "htp_performance_mode", "burst"));
                o.setIntraOpNumThreads(threads);
            }
            default -> throw new OrtException("backend " + b + " is not an ONNX Runtime backend");
        }
        return o;
    }

    public static OrtSession open(String path, Backend b, int threads) throws OrtException {
        try (OrtSession.SessionOptions o = options(b, threads)) {
            return env().createSession(path, o);
        }
    }

    /** Input names, for models whose exports differ slightly. */
    public static boolean hasInput(OrtSession s, String name) throws OrtException {
        Map<String, NodeInfo> in = s.getInputInfo();
        return in.containsKey(name);
    }
}
