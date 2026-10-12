package org.myvoice.accel;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import android.app.ActivityManager;
import android.content.Context;
import android.os.Build;

import org.myvoice.nativebridge.NativeLib;

/** What this phone has: chip, memory, CPU features, Vulkan/OpenCL GPUs and NNAPI devices. */
public final class DeviceInfo {
    public final String model;
    public final String soc;
    public final long ramBytes;
    public final int cores;
    public final String cpuFeatures;
    public final List<String> vulkan;
    public final List<String> opencl;
    public final List<String> nnapi;

    /**
     * @param gpuLists Vulkan, OpenCL and NNAPI device lists as found by the probe process (the
     *                 main process never touches GPU drivers itself); null = not checked yet
     */
    public DeviceInfo(Context c, List<List<String>> gpuLists) {
        model = Build.MANUFACTURER + " " + Build.MODEL;
        soc = (Build.SOC_MANUFACTURER + " " + Build.SOC_MODEL).trim();
        ActivityManager am = (ActivityManager) c.getSystemService(Context.ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
        am.getMemoryInfo(mi);
        ramBytes = mi.totalMem;
        cores = Runtime.getRuntime().availableProcessors();
        cpuFeatures = readCpuFeatures();
        List<String> notYet = List.of("error: not checked yet");
        vulkan = gpuLists != null ? gpuLists.get(0) : notYet;
        opencl = gpuLists != null ? gpuLists.get(1) : notYet;
        nnapi = gpuLists != null ? gpuLists.get(2) : notYet;
    }

    /** In the probe process: ask the drivers (may crash on a broken driver, hence that process). */
    public static List<List<String>> queryAccelerators() {
        if (!NativeLib.loaded()) {
            List<String> e = List.of("error: native library not loaded");
            return List.of(e, e, e);
        }
        return List.of(list(NativeLib.vulkanDevices()), list(NativeLib.openclDevices()), list(NativeLib.nnapiDevices()));
    }

    private static List<String> list(String[] a) {
        List<String> out = new ArrayList<>();
        if (a != null) {
            for (String s : a) {
                out.add(s);
            }
        }
        return out;
    }

    static String readCpuFeatures() {
        File f = new File("/proc/cpuinfo");
        try (BufferedReader r = new BufferedReader(new FileReader(f))) {
            String line;
            while ((line = r.readLine()) != null) {
                if (line.toLowerCase(Locale.ROOT).startsWith("features")) {
                    return line.substring(line.indexOf(':') + 1).trim();
                }
            }
        } catch (IOException ignored) {
        }
        return "";
    }

    public boolean hasCpuFeature(String f) {
        for (String x : cpuFeatures.split("\\s+")) {
            if (x.equals(f)) {
                return true;
            }
        }
        return false;
    }

    public double ramGb() {
        return ramBytes / (1024.0 * 1024 * 1024);
    }

    /** Human-readable summary lines for the setup and acceleration screens. */
    public Map<String, String> summary() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("Phone", model);
        m.put("Chip", soc.isEmpty() ? "unknown" : soc);
        m.put("Memory", String.format(Locale.ROOT, "%.1f GB", ramGb()));
        String isa = Build.SUPPORTED_ABIS.length > 0 && Build.SUPPORTED_ABIS[0].startsWith("x86") ? "x86-64"
                : hasCpuFeature("i8mm") ? "ARM: dot product + int8 matmul"
                : hasCpuFeature("asimddp") ? "ARM: dot product (ARMv8.2)" : "ARM: basic NEON";
        m.put("CPU", cores + " cores · " + isa);
        m.put("Vulkan", describe(vulkan));
        m.put("OpenCL", describe(opencl));
        m.put("NNAPI", describe(nnapi));
        return m;
    }

    private static String describe(List<String> l) {
        if (l.isEmpty()) {
            return "none";
        }
        if (l.size() == 1 && l.get(0).startsWith("error: ")) {
            return "unavailable (" + l.get(0).substring(7) + ")";
        }
        List<String> names = new ArrayList<>();
        for (String s : l) {
            names.add(s.split("\t")[0] + (s.split("\t").length > 1 ? " (" + s.split("\t")[1] + ")" : ""));
        }
        return String.join(", ", names);
    }
}
