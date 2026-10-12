package org.myvoice.accel;

/** Where a model runs. */
public enum Backend {
    QNN("NPU (Qualcomm QNN)"),
    NNAPI("NNAPI (GPU/DSP/NPU)"),
    VULKAN("GPU (Vulkan)"),
    OPENCL("GPU (OpenCL)"),
    XNNPACK("CPU (XNNPACK)"),
    CPU("CPU"),
    SYSTEM("System voice engine");

    public final String label;

    Backend(String label) {
        this.label = label;
    }

    public boolean isAccelerator() {
        return this == QNN || this == NNAPI || this == VULKAN || this == OPENCL;
    }
}
