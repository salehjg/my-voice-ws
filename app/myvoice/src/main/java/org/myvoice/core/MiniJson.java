package org.myvoice.core;

import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A small streaming JSON reader: objects become Maps, arrays Lists, numbers Long/Double. Used for
 * multi-megabyte files (tokenizer.json) where org.json would build far more objects, and in code
 * that also runs in plain-JVM unit tests.
 */
public final class MiniJson {
    private final Reader r;
    private int c;

    public MiniJson(Reader r) throws IOException {
        this.r = r;
        c = r.read();
    }

    private void skip() throws IOException {
        while (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
            c = r.read();
        }
    }

    private void expect(int ch) throws IOException {
        skip();
        if (c != ch) {
            throw new IOException("expected '" + (char) ch + "', got '" + (char) c + "'");
        }
        c = r.read();
    }

    public Object parse() throws IOException {
        skip();
        switch (c) {
            case '{': {
                Map<String, Object> m = new HashMap<>();
                c = r.read();
                skip();
                if (c == '}') {
                    c = r.read();
                    return m;
                }
                while (true) {
                    skip();
                    String key = string();
                    expect(':');
                    m.put(key, parse());
                    skip();
                    if (c == ',') {
                        c = r.read();
                        continue;
                    }
                    expect('}');
                    return m;
                }
            }
            case '[': {
                List<Object> l = new ArrayList<>();
                c = r.read();
                skip();
                if (c == ']') {
                    c = r.read();
                    return l;
                }
                while (true) {
                    l.add(parse());
                    skip();
                    if (c == ',') {
                        c = r.read();
                        continue;
                    }
                    expect(']');
                    return l;
                }
            }
            case '"':
                return string();
            case 't':
                word("true");
                return Boolean.TRUE;
            case 'f':
                word("false");
                return Boolean.FALSE;
            case 'n':
                word("null");
                return null;
            default:
                return number();
        }
    }

    private void word(String w) throws IOException {
        for (int i = 0; i < w.length(); i++) {
            if (c != w.charAt(i)) {
                throw new IOException("bad literal");
            }
            c = r.read();
        }
    }

    private Number number() throws IOException {
        StringBuilder b = new StringBuilder();
        while (c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' || (c >= '0' && c <= '9')) {
            b.append((char) c);
            c = r.read();
        }
        String s = b.toString();
        if (s.isEmpty()) {
            throw new IOException("unexpected '" + (char) c + "'");
        }
        if (s.contains(".") || s.contains("e") || s.contains("E")) {
            return Double.parseDouble(s);
        }
        return Long.parseLong(s);
    }

    private String string() throws IOException {
        if (c != '"') {
            throw new IOException("expected a string");
        }
        StringBuilder b = new StringBuilder();
        c = r.read();
        while (c != '"') {
            if (c < 0) {
                throw new IOException("unterminated string");
            }
            if (c == '\\') {
                c = r.read();
                switch (c) {
                    case 'n' -> b.append('\n');
                    case 't' -> b.append('\t');
                    case 'r' -> b.append('\r');
                    case 'b' -> b.append('\b');
                    case 'f' -> b.append('\f');
                    case 'u' -> {
                        char[] hex = new char[4];
                        for (int i = 0; i < 4; i++) {
                            hex[i] = (char) r.read();
                        }
                        b.append((char) Integer.parseInt(new String(hex), 16));
                    }
                    default -> b.append((char) c);
                }
            } else {
                b.append((char) c);
            }
            c = r.read();
        }
        c = r.read();
        return b.toString();
    }
}
