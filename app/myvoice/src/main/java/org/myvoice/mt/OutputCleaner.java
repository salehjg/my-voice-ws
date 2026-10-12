package org.myvoice.mt;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/** Strip reasoning blocks, labels, notes and wrapping quotes from a model reply (port of clean_output). */
public final class OutputCleaner {
    private static final Pattern THINK = Pattern.compile("<think>.*?</think>", Pattern.DOTALL);
    private static final Pattern LABEL = Pattern.compile(
            "^\\s*(?:translation|traduzione|english|italian|inglese|italiano|output|answer|risposta)"
            + "\\s*(?:\\([^)]*\\))?\\s*:\\s*", Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE);
    private static final Pattern NOTE = Pattern.compile(
            "^\\s*[(\\[]?\\s*(?:note|nota|n\\.b\\.|explanation|spiegazione)\\b",
            Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE);
    private static final String[][] QUOTE_PAIRS = {{"\"", "\""}, {"“", "”"}, {"«", "»"}, {"„", "“"}, {"'", "'"}};
    private static final Pattern SPACES = Pattern.compile("\\s+");

    private OutputCleaner() {
    }

    public static String clean(String raw, String source) {
        String text = THINK.matcher(raw).replaceAll("");
        int end = text.lastIndexOf("</think>");
        if (end >= 0) {
            text = text.substring(end + "</think>".length());
        }
        int open = text.indexOf("<think>");
        if (open >= 0) {   // unterminated reasoning (ran out of tokens): nothing usable after it
            text = text.substring(0, open);
        }
        for (String stop : Prompts.STOP_STRINGS) {
            int at = text.indexOf(stop);
            if (at >= 0) {
                text = text.substring(0, at);
            }
        }
        List<String> lines = new ArrayList<>();
        for (String line : text.strip().split("\\R", -1)) {
            if (!lines.isEmpty() && NOTE.matcher(line).find()) {
                break;
            }
            if (!line.isBlank()) {
                lines.add(line.strip());
            }
        }
        text = String.join(" ", lines);
        text = LABEL.matcher(text).replaceFirst("").strip();
        String src = source.strip();
        for (String[] q : QUOTE_PAIRS) {
            if (text.length() >= 2 && text.startsWith(q[0]) && text.endsWith(q[1]) && !src.startsWith(q[0])) {
                text = text.substring(q[0].length(), text.length() - q[1].length()).strip();
                break;
            }
        }
        return SPACES.matcher(text).replaceAll(" ").strip();
    }
}
