package org.myvoice.lang;

/** A language of the pair: ISO 639-1 code ("it"), English name ("Italian"), locale code ("it-IT"). */
public record Language(String code, String name, String regionCode) {
}
