package org.myvoice.vad;

import java.nio.FloatBuffer;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.myvoice.accel.Backend;
import org.myvoice.accel.OrtSessions;

/**
 * Silero VAD (the laptop's own 16 kHz ONNX model, bundled in the APK). Runs on one CPU thread:
 * it is tiny and called every 32 ms, so an accelerator would only add latency.
 *
 * <p>Like silero-vad's own ONNX wrapper, each 512-sample frame is fed with the previous frame's
 * last 64 samples in front of it, and the recurrent state is carried between calls.
 */
public final class SileroVad implements FrameVad {
    private static final int CONTEXT = 64;
    private final OrtEnvironment env = OrtSessions.env();
    private final OrtSession session;
    private float[] state = new float[2 * 128];
    private final float[] context = new float[CONTEXT];

    public SileroVad(byte[] model) throws OrtException {
        try (OrtSession.SessionOptions o = OrtSessions.options(Backend.CPU, 1)) {
            o.setInterOpNumThreads(1);
            session = env.createSession(model, o);
        }
    }

    @Override
    public int frameSize() {
        return 512;
    }

    @Override
    public synchronized float prob(float[] frame) {
        float[] x = new float[CONTEXT + frame.length];
        System.arraycopy(context, 0, x, 0, CONTEXT);
        System.arraycopy(frame, 0, x, CONTEXT, frame.length);
        try (OnnxTensor input = OnnxTensor.createTensor(env, FloatBuffer.wrap(x), new long[] {1, x.length});
             OnnxTensor st = OnnxTensor.createTensor(env, FloatBuffer.wrap(state), new long[] {2, 1, 128});
             OnnxTensor sr = OnnxTensor.createTensor(env, 16000L);
             OrtSession.Result r = session.run(Map.of("input", input, "state", st, "sr", sr))) {
            float p = ((float[][]) r.get(0).getValue())[0][0];
            FloatBuffer ns = ((OnnxTensor) r.get(1)).getFloatBuffer();
            float[] next = new float[ns.remaining()];
            ns.get(next);
            state = next;
            System.arraycopy(x, x.length - CONTEXT, context, 0, CONTEXT);
            return p;
        } catch (OrtException e) {
            return 0f;
        }
    }

    @Override
    public synchronized void reset() {
        state = new float[2 * 128];
        java.util.Arrays.fill(context, 0f);
    }

    @Override
    public void close() {
        try {
            session.close();
        } catch (OrtException ignored) {
        }
    }
}
