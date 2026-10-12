package org.myvoice.mt;

/** Text translation between the two languages of the pair. */
public interface Translator extends AutoCloseable {
    String translate(String text, String srcCode, String tgtCode) throws Exception;

    /** What it runs on, e.g. "TranslateGemma 4B · CPU ×4". */
    String describe();

    @Override
    default void close() {
    }
}
