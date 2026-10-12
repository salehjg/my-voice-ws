package org.myvoice.accel;

import java.util.List;

/** A step of the pipeline and the backends tried for it, best first. */
public enum Stage {
    VAD("Voice detection", List.of(Backend.CPU)),
    ASR("Speech → text", List.of(Backend.VULKAN, Backend.OPENCL, Backend.CPU)),
    MT_LLM("Translation", List.of(Backend.VULKAN, Backend.OPENCL, Backend.CPU)),
    MT_ONNX("Translation", List.of(Backend.QNN, Backend.NNAPI, Backend.XNNPACK, Backend.CPU)),
    TTS_PIPER("Voice", List.of(Backend.NNAPI, Backend.XNNPACK, Backend.CPU)),
    TTS_SYSTEM("Voice", List.of(Backend.SYSTEM));

    public final String label;
    public final List<Backend> candidates;

    Stage(String label, List<Backend> candidates) {
        this.label = label;
        this.candidates = candidates;
    }
}
