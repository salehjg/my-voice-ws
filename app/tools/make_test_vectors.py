"""Reference values for the app's Java code, computed with the laptop's Python tools.

    source setvars.all.sh; conda activate myvoice
    python app/tools/make_test_vectors.py OPUS_DIR OUT.json

OPUS_DIR holds the two Opus-MT exports (it-en/, en-it/ as in assets/models.json). Writes:
  tokens      sentence -> ids, from the `tokenizers` library (the Java Unigram tokenizer must match)
  translations sentence -> greedy Opus-MT translation, decoded here with onnxruntime independently
  silero      speech probability per 512-sample frame of assets/probe_it.wav (silero-vad's own wrapper)
  gemma_prompt TranslateGemma prompt rendered from the GGUF's own chat template
  piper       espeak clauses for a sentence and the phoneme ids Piper itself makes from them
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
opus, out = Path(sys.argv[1]), Path(sys.argv[2])
SENTENCES = {
    "it-en": ["Buongiorno, vorrei prenotare un tavolo per due persone stasera alle otto.",
              "Mi scusi, sa dirmi dov'è la stazione dei treni?",
              "Perfetto, grazie mille. E quanto costa il biglietto?"],
    "en-it": ["Excuse me, could you tell me how to get to the train station?",
              "I think it leaves at 11.30. You can buy the ticket on board.",
              "Very well, thank you."],
}


def load_tokenizer(d: Path) -> Tokenizer:
    """The exports have an empty "Precompiled" normaliser that `tokenizers` rejects (transformers.js
    accepts it); use NFKC like the Java tokenizer (and Marian's SentencePiece) does."""
    spec = json.loads((d / "tokenizer.json").read_text())
    spec["normalizer"] = {"type": "NFKC"}
    return Tokenizer.from_str(json.dumps(spec))


def greedy(d: Path, text: str) -> str:
    tok = load_tokenizer(d)
    cfg = json.loads((d / "config.json").read_text())
    enc = ort.InferenceSession(str(d / "onnx/encoder_model_quantized.onnx"), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(d / "onnx/decoder_model_merged_quantized.onnx"), providers=["CPUExecutionProvider"])
    ids = np.array([tok.encode(text).ids], dtype=np.int64)
    mask = np.ones_like(ids)
    hidden = enc.run(None, {"input_ids": ids, "attention_mask": mask})[0]
    heads, dim = cfg["decoder_attention_heads"], cfg["d_model"] // cfg["decoder_attention_heads"]
    past_names = [i.name for i in dec.get_inputs() if i.name.startswith("past_key_values.")]
    out_names = [o.name for o in dec.get_outputs()]
    past = {n: np.zeros((1, heads, 0, dim), dtype=np.float32) for n in past_names}
    last, result, cached = cfg["decoder_start_token_id"], [], False
    for _ in range(min(512, 2 * ids.shape[1] + 20)):
        feed = {"input_ids": np.array([[last]], dtype=np.int64), "encoder_attention_mask": mask,
                "encoder_hidden_states": hidden, "use_cache_branch": np.array([cached]), **past}
        outs = dict(zip(out_names, dec.run(None, feed)))
        logits = outs["logits"][0, -1].copy()
        logits[cfg["pad_token_id"]] = -np.inf
        nxt = int(np.argmax(logits))
        if nxt == cfg["eos_token_id"]:
            break
        result.append(nxt)
        # the encoder key/values come from the first step only: in the cached branch the model's
        # encoder outputs are placeholders
        past = {n: (outs[n.replace("past_key_values.", "present.")] if (".decoder." in n or not cached) else past[n])
                for n in past_names}
        last, cached = nxt, True
    return tok.decode(result, skip_special_tokens=True)


golden = {"tokens": {}, "translations": {}}
for direction, sentences in SENTENCES.items():
    d = opus / direction
    tok = load_tokenizer(d)
    for s in sentences:
        golden["tokens"][f"{direction}|{s}"] = tok.encode(s).ids
        parts = [x for x in re.split(r"(?<=[.!?…])\s+(?=\S)", s.strip()) if x.strip()]
        golden["translations"][f"{direction}|{s}"] = " ".join(greedy(d, x) for x in parts)
        print(direction, "|", s, "->", golden["translations"][f"{direction}|{s}"])

from silero_vad.utils_vad import OnnxWrapper  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from myvoice.audio import read_audio  # noqa: E402

vad = OnnxWrapper(str(APP / "myvoice/src/main/assets/silero_vad.onnx"), force_onnx_cpu=True)
audio = read_audio(APP / "myvoice/src/main/assets/probe_it.wav", 16000)
import torch  # noqa: E402

golden["silero"] = [float(vad(torch.from_numpy(audio[i:i + 512].copy()), 16000))
                    for i in range(0, len(audio) - 511, 512)]

from llama_cpp import Llama  # noqa: E402
from myvoice.translate import render_chat_template, translategemma_messages  # noqa: E402

llm = Llama(str(ROOT / "models/llm/translategemma-4b-it.Q6_K.gguf"), vocab_only=True, verbose=False)
template = llm.metadata["tokenizer.chat_template"]
# like the laptop's Translator.build_prompt: region codes first, then plain codes
text = "Buongiorno, vorrei prenotare un tavolo per due persone stasera alle otto."
for codes in (("it-IT", "en-US"), ("it", "en")):
    try:
        golden["gemma_prompt"] = {"text": text, "codes": list(codes),
                                  "prompt": render_chat_template(template, translategemma_messages(text, *codes))}
        break
    except Exception as e:  # noqa: BLE001
        print("template rejects", codes, ":", e)
from piper import PiperVoice, espeakbridge  # noqa: E402

golden["piper"] = []
for rel, text in (("it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx", "Buongiorno, come sta? Il treno parte alle nove: non tardi!"),
                  ("en/en_US/ryan/high/en_US-ryan-high.onnx", "Excuse me, could you tell me how to get to the station? It's 5 o'clock.")):
    onnx = ROOT / "models/piper" / rel
    voice = PiperVoice.load(str(onnx))
    sentences = voice.phonemize(text)   # initialises espeak with Piper's data and sets the voice
    clauses = [f"{ph}\u0001{t}\u0001{'1' if e else '0'}" for ph, t, e in espeakbridge.get_phonemes(text)]
    golden["piper"].append({"config": str(onnx) + ".json", "text": text, "clauses": clauses,
                            "ids": [voice.phonemes_to_ids(s) for s in sentences]})

out.write_text(json.dumps(golden, ensure_ascii=False, indent=1))
print("wrote", out, "silero frames:", len(golden["silero"]))
