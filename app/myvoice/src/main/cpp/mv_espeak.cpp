// espeak-ng phonemes for Piper voices: a port of Piper's own espeakbridge.c (piper1-gpl).

#include "mv_native.h"

#include <espeak-ng/speak_lib.h>

#define CLAUSE_INTONATION_FULL_STOP 0x00000000
#define CLAUSE_INTONATION_COMMA 0x00001000
#define CLAUSE_INTONATION_QUESTION 0x00002000
#define CLAUSE_INTONATION_EXCLAMATION 0x00003000
#define CLAUSE_TYPE_CLAUSE 0x00040000
#define CLAUSE_TYPE_SENTENCE 0x00080000
#define CLAUSE_PERIOD (40 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_COMMA (20 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE)
#define CLAUSE_QUESTION (40 | CLAUSE_INTONATION_QUESTION | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_EXCLAMATION (45 | CLAUSE_INTONATION_EXCLAMATION | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_COLON (30 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_CLAUSE)
#define CLAUSE_SEMICOLON (30 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE)

int mv_espeak_init(const std::string & data_dir) {
    return espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data_dir.c_str(), 0) < 0 ? -1 : 0;
}

int mv_espeak_set_voice(const std::string & voice) {
    return espeak_SetVoiceByName(voice.c_str()) == EE_OK ? 0 : -1;
}

std::vector<std::string> mv_espeak_phonemize(const std::string & text) {
    std::vector<std::string> out;
    const char * p = text.c_str();
    while (p != nullptr) {
        int terminator = 0;
        const char * phonemes = espeak_TextToPhonemesWithTerminator((const void **) &p, espeakCHARS_AUTO,
                                                                    espeakPHONEMES_IPA, &terminator);
        terminator &= 0x000FFFFF;
        const char * t = "";
        if (terminator == CLAUSE_PERIOD) {
            t = ".";
        } else if (terminator == CLAUSE_QUESTION) {
            t = "?";
        } else if (terminator == CLAUSE_EXCLAMATION) {
            t = "!";
        } else if (terminator == CLAUSE_COMMA) {
            t = ",";
        } else if (terminator == CLAUSE_COLON) {
            t = ":";
        } else if (terminator == CLAUSE_SEMICOLON) {
            t = ";";
        }
        bool eos = (terminator & CLAUSE_TYPE_SENTENCE) == CLAUSE_TYPE_SENTENCE;
        out.push_back(std::string(phonemes ? phonemes : "") + "\x01" + t + "\x01" + (eos ? "1" : "0"));
    }
    return out;
}
