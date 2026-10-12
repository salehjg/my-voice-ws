package org.myvoice.mt;

import java.util.List;

import org.myvoice.lang.Language;
import org.myvoice.lang.Languages;

/** LLM prompts (port of translate.py): TranslateGemma's official prompt and a chat "interpreter" prompt. */
public final class Prompts {
    /**
     * Exactly what TranslateGemma's own chat template (in the GGUF) renders, which is what the
     * laptop sends: plain language codes, and a line break before "Produce only".
     */
    static final String TRANSLATEGEMMA_INSTRUCTION =
            "You are a professional %1$s (%2$s) to %3$s (%4$s) translator. "
            + "Your goal is to accurately convey the meaning and nuances of the original %1$s text "
            + "while adhering to %3$s grammar, vocabulary, and cultural sensitivities.\n"
            + "Produce only the %3$s translation, without any additional explanations or commentary. "
            + "Please translate the following %1$s text into %3$s:\n\n\n%5$s";

    static final String CHAT_SYSTEM_PROMPT =
            "You are a professional simultaneous interpreter. Translate the user's %1$s speech "
            + "into natural, idiomatic spoken %2$s. Preserve meaning, tone and register (formal "
            + "Lei / informal tu), and keep names, numbers and units exact. The text comes from speech "
            + "recognition, so it may lack punctuation or contain small recognition errors: translate "
            + "what the speaker meant. Reply with the %2$s translation only: no quotes, notes, "
            + "alternatives or explanations.";

    /** Strings that end a reply (the turn markers of the supported model families). */
    public static final List<String> STOP_STRINGS =
            List.of("<end_of_turn>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<eos>");

    public record Exchange(String srcLang, String srcText, String tgtLang, String tgtText) {
    }

    private Prompts() {
    }

    public static String translateGemmaInstruction(String text, Language src, Language tgt) {
        return String.format(TRANSLATEGEMMA_INSTRUCTION, src.name(), src.code(), tgt.name(), tgt.code(), text);
    }

    /** Gemma-3 turn format with the official instruction (BOS is added by the tokenizer). */
    public static String translateGemma(String text, Language src, Language tgt) {
        return "<start_of_turn>user\n" + translateGemmaInstruction(text, src, tgt) + "<end_of_turn>\n<start_of_turn>model\n";
    }

    public static String chatSystem(Language src, Language tgt, List<Exchange> history) {
        StringBuilder system = new StringBuilder(String.format(CHAT_SYSTEM_PROMPT, src.name(), tgt.name()));
        if (!history.isEmpty()) {
            system.append("\n\nRecent conversation, for context only (do not translate it again):\n");
            for (int i = 0; i < history.size(); i++) {
                Exchange ex = history.get(i);
                if (i > 0) {
                    system.append('\n');
                }
                system.append("- ").append(Languages.name(ex.srcLang())).append(": ").append(ex.srcText()).append('\n');
                system.append("  ").append(Languages.name(ex.tgtLang())).append(": ").append(ex.tgtText());
            }
        }
        return system.toString();
    }

    /** Gemma chat format: Gemma has no system role, so its template puts the system text in front of the first user turn. */
    public static String gemmaChat(String text, Language src, Language tgt, List<Exchange> history) {
        return "<start_of_turn>user\n" + chatSystem(src, tgt, history) + "\n\n" + text
                + "<end_of_turn>\n<start_of_turn>model\n";
    }

    public static int maxTokensFor(String text, int maxNewTokens) {
        String t = text.strip();
        int words = t.isEmpty() ? 0 : t.split("\\s+").length;
        return Math.min(maxNewTokens, 64 + 4 * words);
    }
}
