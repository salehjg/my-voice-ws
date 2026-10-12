package org.myvoice.core;

import java.util.Locale;
import java.util.Map;

/** Just enough JSON writing for the session log (keeps the core free of Android classes). */
public final class Json {
    private Json() {
    }

    public static String quote(String s) {
        StringBuilder b = new StringBuilder(s.length() + 2).append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> b.append("\\\"");
                case '\\' -> b.append("\\\\");
                case '\n' -> b.append("\\n");
                case '\r' -> b.append("\\r");
                case '\t' -> b.append("\\t");
                default -> {
                    if (c < 0x20) {
                        b.append(String.format(Locale.ROOT, "\\u%04x", (int) c));
                    } else {
                        b.append(c);
                    }
                }
            }
        }
        return b.append('"').toString();
    }

    public static String number(double v) {
        if (v == Math.rint(v) && Math.abs(v) < 1e15) {
            return String.format(Locale.ROOT, "%.1f", v);
        }
        return String.format(Locale.ROOT, "%.4f", v).replaceAll("0+$", "");
    }

    public static String object(Map<String, ?> fields) {
        StringBuilder b = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, ?> e : fields.entrySet()) {
            if (!first) {
                b.append(", ");
            }
            first = false;
            b.append(quote(e.getKey())).append(": ").append(value(e.getValue()));
        }
        return b.append('}').toString();
    }

    @SuppressWarnings("unchecked")
    private static String value(Object v) {
        if (v == null) {
            return "null";
        }
        if (v instanceof String s) {
            return quote(s);
        }
        if (v instanceof Boolean bo) {
            return bo.toString();
        }
        if (v instanceof Number n) {
            return number(n.doubleValue());
        }
        if (v instanceof Map<?, ?> m) {
            return object((Map<String, ?>) m);
        }
        return quote(v.toString());
    }
}
