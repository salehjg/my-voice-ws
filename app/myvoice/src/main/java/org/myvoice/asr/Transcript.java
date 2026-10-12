package org.myvoice.asr;

/** One recognised utterance; {@code languageProb} is renormalised over the pair (1.0 when forced). */
public record Transcript(String text, String language, float languageProb) {
}
