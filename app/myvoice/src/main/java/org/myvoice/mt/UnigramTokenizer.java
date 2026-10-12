package org.myvoice.mt;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.myvoice.core.MiniJson;

/**
 * SentencePiece-style Unigram tokenizer read from a Hugging Face {@code tokenizer.json} (the
 * format of the Opus-MT exports): split on whitespace, prefix each word with "▁", then pick the
 * most probable segmentation of each word into vocabulary pieces (Viterbi). Token id = position
 * in the vocabulary list, so source and target share ids (Marian's shared vocabulary).
 */
public final class UnigramTokenizer {
    public static final char SPACE = '▁';

    private final String[] pieces;
    private final float[] scores;
    private final Map<String, Integer> ids = new HashMap<>();
    private final int maxPieceLength;
    private final int unkId;
    public final int eosId;
    public final int padId;

    public UnigramTokenizer(List<String> pieces, List<Float> scores, int unkId, int eosId, int padId) {
        this.pieces = pieces.toArray(new String[0]);
        this.scores = new float[scores.size()];
        int maxLen = 1;
        for (int i = 0; i < this.pieces.length; i++) {
            this.scores[i] = scores.get(i);
            ids.putIfAbsent(this.pieces[i], i);
            maxLen = Math.max(maxLen, this.pieces[i].length());
        }
        this.maxPieceLength = maxLen;
        this.unkId = unkId;
        this.eosId = eosId;
        this.padId = padId;
    }

    public int vocabSize() {
        return pieces.length;
    }

    /** Load from tokenizer.json (model.vocab = [[piece, score], ...]; special ids from added_tokens). */
    public static UnigramTokenizer fromJson(InputStream in) throws IOException {
        try (Reader r = new InputStreamReader(in, StandardCharsets.UTF_8)) {
            Object root = new MiniJson(r).parse();
            Map<?, ?> model = (Map<?, ?>) ((Map<?, ?>) root).get("model");
            if (!"Unigram".equals(model.get("type"))) {
                throw new IOException("not a Unigram tokenizer: " + model.get("type"));
            }
            List<String> pieces = new ArrayList<>();
            List<Float> scores = new ArrayList<>();
            for (Object o : (List<?>) model.get("vocab")) {
                List<?> pair = (List<?>) o;
                pieces.add((String) pair.get(0));
                scores.add(((Number) pair.get(1)).floatValue());
            }
            Map<String, Integer> special = new HashMap<>();
            Object added = ((Map<?, ?>) root).get("added_tokens");
            if (added instanceof List<?> list) {
                for (Object o : list) {
                    Map<?, ?> t = (Map<?, ?>) o;
                    special.put((String) t.get("content"), ((Number) t.get("id")).intValue());
                }
            }
            int unk = special.getOrDefault("<unk>", ((Number) model.get("unk_id")).intValue());
            int eos = special.getOrDefault("</s>", 0);
            int pad = special.getOrDefault("<pad>", pieces.size() - 1);
            return new UnigramTokenizer(pieces, scores, unk, eos, pad);
        }
    }

    /** Text -> ids, ending with {@code </s>}. */
    public List<Integer> encode(String text) {
        String norm = Normalizer.normalize(text, Normalizer.Form.NFKC);
        List<Integer> out = new ArrayList<>();
        for (String word : norm.strip().split("\\s+")) {
            if (!word.isEmpty()) {
                encodeWord(SPACE + word, out);
            }
        }
        out.add(eosId);
        return out;
    }

    private void encodeWord(String w, List<Integer> out) {
        int n = w.length();
        double[] best = new double[n + 1];
        int[] from = new int[n + 1];
        int[] tok = new int[n + 1];
        java.util.Arrays.fill(best, Double.NEGATIVE_INFINITY);
        best[0] = 0;
        for (int end = 1; end <= n; end++) {
            for (int start = Math.max(0, end - maxPieceLength); start < end; start++) {
                if (best[start] == Double.NEGATIVE_INFINITY) {
                    continue;
                }
                Integer id = ids.get(w.substring(start, end));
                if (id == null) {
                    continue;
                }
                double sc = best[start] + scores[id];
                if (sc > best[end]) {
                    best[end] = sc;
                    from[end] = start;
                    tok[end] = id;
                }
            }
            if (best[end] == Double.NEGATIVE_INFINITY && best[end - 1] != Double.NEGATIVE_INFINITY) {
                // unknown character: one <unk> covering it (heavily penalised, like SentencePiece)
                best[end] = best[end - 1] - 10.0;
                from[end] = end - 1;
                tok[end] = unkId;
            }
        }
        List<Integer> rev = new ArrayList<>();
        for (int pos = n; pos > 0; pos = from[pos]) {
            rev.add(tok[pos]);
        }
        int last = -1;
        for (int i = rev.size() - 1; i >= 0; i--) {
            int id = rev.get(i);
            if (id == unkId && last == unkId) {
                continue;   // runs of unknown characters become one <unk>
            }
            out.add(id);
            last = id;
        }
    }

    /** Ids -> text, without special tokens. */
    public String decode(List<Integer> tokens) {
        StringBuilder b = new StringBuilder();
        for (int id : tokens) {
            if (id == eosId || id == padId || id == unkId || id < 0 || id >= pieces.length) {
                continue;
            }
            b.append(pieces[id]);
        }
        return b.toString().replace(SPACE, ' ').strip();
    }
}
