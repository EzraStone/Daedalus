package dev.daedalus.harness;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A tiny JSON reader for the harness protocol.
 *
 * <p>Deliberately vendored rather than pulled from Gson: the harness has to be
 * droppable into a server directory without dependency resolution, and the
 * protocol is five fields wide.
 */
public final class Json {
    public static final class Obj {
        private final Map<String, Object> fields = new LinkedHashMap<>();

        public String string(String key) {
            Object v = fields.get(key);
            if (!(v instanceof String)) {
                throw new IllegalArgumentException("missing string field " + key);
            }
            return (String) v;
        }

        @SuppressWarnings("unchecked")
        public List<int[]> positions(String key) {
            Object v = fields.get(key);
            if (!(v instanceof List)) {
                throw new IllegalArgumentException("missing array field " + key);
            }
            List<int[]> out = new ArrayList<>();
            for (Object item : (List<Object>) v) {
                List<Object> triple = (List<Object>) item;
                int[] pos = new int[triple.size()];
                for (int i = 0; i < pos.length; i++) {
                    pos[i] = (int) Math.round((Double) triple.get(i));
                }
                out.add(pos);
            }
            return out;
        }

        void put(String key, Object value) {
            fields.put(key, value);
        }
    }

    public static String quote(String s) {
        StringBuilder sb = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"': sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        return sb.append('"').toString();
    }

    public static Obj parse(String text) {
        Parser p = new Parser(text);
        p.skipWhitespace();
        Object value = p.value();
        if (!(value instanceof Obj)) {
            throw new IllegalArgumentException("expected a JSON object");
        }
        return (Obj) value;
    }

    private static final class Parser {
        private final String src;
        private int at;

        Parser(String src) {
            this.src = src;
        }

        void skipWhitespace() {
            while (at < src.length() && Character.isWhitespace(src.charAt(at))) {
                at++;
            }
        }

        Object value() {
            skipWhitespace();
            char c = src.charAt(at);
            if (c == '{') {
                return object();
            }
            if (c == '[') {
                return array();
            }
            if (c == '"') {
                return string();
            }
            if (src.startsWith("true", at)) {
                at += 4;
                return Boolean.TRUE;
            }
            if (src.startsWith("false", at)) {
                at += 5;
                return Boolean.FALSE;
            }
            if (src.startsWith("null", at)) {
                at += 4;
                return null;
            }
            return number();
        }

        Obj object() {
            Obj obj = new Obj();
            at++; // {
            skipWhitespace();
            if (src.charAt(at) == '}') {
                at++;
                return obj;
            }
            while (true) {
                skipWhitespace();
                String key = string();
                skipWhitespace();
                at++; // :
                obj.put(key, value());
                skipWhitespace();
                char c = src.charAt(at++);
                if (c == '}') {
                    return obj;
                }
                if (c != ',') {
                    throw new IllegalArgumentException("expected , or } at " + at);
                }
            }
        }

        List<Object> array() {
            List<Object> out = new ArrayList<>();
            at++; // [
            skipWhitespace();
            if (src.charAt(at) == ']') {
                at++;
                return out;
            }
            while (true) {
                out.add(value());
                skipWhitespace();
                char c = src.charAt(at++);
                if (c == ']') {
                    return out;
                }
                if (c != ',') {
                    throw new IllegalArgumentException("expected , or ] at " + at);
                }
            }
        }

        String string() {
            StringBuilder sb = new StringBuilder();
            at++; // opening quote
            while (true) {
                char c = src.charAt(at++);
                if (c == '"') {
                    return sb.toString();
                }
                if (c == '\\') {
                    char esc = src.charAt(at++);
                    switch (esc) {
                        case 'n': sb.append('\n'); break;
                        case 'r': sb.append('\r'); break;
                        case 't': sb.append('\t'); break;
                        case 'u':
                            sb.append((char) Integer.parseInt(src.substring(at, at + 4), 16));
                            at += 4;
                            break;
                        default: sb.append(esc);
                    }
                } else {
                    sb.append(c);
                }
            }
        }

        Double number() {
            int start = at;
            while (at < src.length() && "-+.eE0123456789".indexOf(src.charAt(at)) >= 0) {
                at++;
            }
            return Double.valueOf(src.substring(start, at));
        }
    }
}
