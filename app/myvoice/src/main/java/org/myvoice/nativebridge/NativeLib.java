package org.myvoice.nativebridge;

/**
 * JNI entry points of {@code libmyvoice_jni.so} (src/main/cpp): ggml backends, whisper.cpp,
 * llama.cpp, espeak-ng and the device probes. Handles are opaque native pointers.
 */
public final class NativeLib {
    private static Throwable loadError;

    static {
        try {
            System.loadLibrary("myvoice_jni");
        } catch (Throwable t) {   // e.g. a build without the native part (-PnoNative)
            loadError = t;
        }
    }

    private NativeLib() {
    }

    public static boolean loaded() {
        return loadError == null;
    }

    public static String loadError() {
        return loadError == null ? "" : loadError.toString();
    }

    // ------------------------------------------------------------------ ggml backends

    /**
     * Load the ggml backend modules found in {@code libDir}: always the best CPU variant for this
     * CPU, plus "vulkan" / "opencl" when listed in {@code gpuBackends} (comma separated). Returns
     * one line per device: "name\tdescription\ttype\tbackend".
     */
    public static native String[] ggmlLoadBackends(String libDir, String gpuBackends);

    /** Features of the CPU variant ggml picked, e.g. "NEON DOTPROD FP16". */
    public static native String ggmlCpuFeatures();

    // ------------------------------------------------------------------ whisper.cpp

    /** @param device ggml device name to run on ("" = CPU) */
    public static native long whisperInit(String modelPath, String device, boolean flashAttn);

    /**
     * Transcribe 16 kHz mono audio. {@code language} "" = detect; {@code langProbs} (size
     * whisperLangCount) receives the detection probabilities when detecting. Returns
     * "lang\ttext", or null on failure.
     */
    public static native String whisperTranscribe(long ctx, float[] audio, String language, int threads,
                                                  String initialPrompt, int audioCtx, float[] langProbs);

    public static native int whisperLangId(String code);

    public static native int whisperLangCount();

    public static native void whisperFree(long ctx);

    // ------------------------------------------------------------------ llama.cpp

    /** @param device ggml device name ("" = CPU); {@code gpuLayers} -1 = all */
    public static native long llamaInit(String modelPath, String device, int gpuLayers, int nCtx, int threads);

    /** Greedy completion of {@code prompt} until a stop string or {@code maxTokens}; null on failure. */
    public static native String llamaComplete(long ctx, String prompt, int maxTokens, String[] stops);

    public static native void llamaFree(long ctx);

    // ------------------------------------------------------------------ espeak-ng (Piper phonemes)

    public static native int espeakInit(String dataDir);

    public static native int espeakSetVoice(String voice);

    /** One entry per clause: phonemes \u0001 terminator \u0001 end-of-sentence ("0"/"1"). */
    public static native String[] espeakPhonemize(String text);

    // ------------------------------------------------------------------ device facts

    /** Vulkan devices as "name\tapiVersion\tdriverVersion\ttype", or a single "error: ..." line. */
    public static native String[] vulkanDevices();

    /** OpenCL platforms/devices as "platform\tdevice\tversion", or a single "error: ..." line. */
    public static native String[] openclDevices();

    /** NNAPI devices as "name\ttype\tversion" (Android 10+). */
    public static native String[] nnapiDevices();
}
