package org.myvoice.mt;

import java.io.File;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;

import org.myvoice.accel.Backend;
import org.myvoice.lang.Language;
import org.myvoice.lang.Languages;
import org.myvoice.nativebridge.NativeLib;

/**
 * LLM translation through llama.cpp (ggml: CPU, Vulkan or OpenCL), greedy like the laptop.
 * TranslateGemma uses Google's official prompt; other chat models (Gemma 3 1B) get the laptop's
 * "interpreter" prompt with the last turns as context.
 */
public final class LlamaTranslator implements Translator {
    public enum Style { TRANSLATEGEMMA, CHAT }

    private static final int MAX_NEW_TOKENS = 384;
    private static final int CONTEXT_TURNS = 2;
    private final long ctx;
    private final Style style;
    private final String name;
    private final Backend backend;
    private final String deviceLabel;
    private final int threads;
    private final Deque<Prompts.Exchange> history = new ArrayDeque<>();

    public LlamaTranslator(File model, Style style, Backend backend, String device, String deviceLabel, int threads) {
        this.ctx = NativeLib.llamaInit(model.getPath(), backend == Backend.CPU ? "" : device,
                backend == Backend.CPU ? 0 : -1, 1024, threads);
        if (ctx == 0) {
            throw new IllegalStateException("llama.cpp could not load " + model.getName() + " on " + backend.label);
        }
        this.style = style;
        this.name = model.getName().replace(".gguf", "");
        this.backend = backend;
        this.deviceLabel = deviceLabel;
        this.threads = threads;
    }

    @Override
    public synchronized String translate(String text, String srcCode, String tgtCode) {
        Language src = Languages.get(srcCode), tgt = Languages.get(tgtCode);
        String prompt = style == Style.TRANSLATEGEMMA
                ? Prompts.translateGemma(text, src, tgt)
                : Prompts.gemmaChat(text, src, tgt, new ArrayList<>(history));
        String raw = NativeLib.llamaComplete(ctx, prompt, Prompts.maxTokensFor(text, MAX_NEW_TOKENS),
                Prompts.STOP_STRINGS.toArray(new String[0]));
        if (raw == null) {
            throw new IllegalStateException("llama.cpp failed to translate");
        }
        String out = OutputCleaner.clean(raw, text);
        if (style == Style.CHAT && !out.isEmpty()) {
            history.addLast(new Prompts.Exchange(srcCode, text, tgtCode, out));
            while (history.size() > CONTEXT_TURNS) {
                history.removeFirst();
            }
        }
        return out;
    }

    @Override
    public String describe() {
        return name + " · " + (backend == Backend.CPU ? "CPU ×" + threads : deviceLabel);
    }

    @Override
    public synchronized void close() {
        NativeLib.llamaFree(ctx);
    }
}
