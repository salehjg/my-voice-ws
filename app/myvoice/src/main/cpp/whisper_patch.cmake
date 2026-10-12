# Writes a patched copy of whisper.cpp/src/whisper.cpp (the submodule itself stays untouched):
#  1. the shortened audio window (audio_ctx) is used for language detection too, not only for
#     transcription, which matters on slow phones;
#  2. the language-detection probabilities of the last run are kept for the caller, so the app
#     can restrict detection to Italian/English without a second detection pass;
#  3. the encoder result from language detection is reused for the transcription of the same
#     window (whisper.cpp otherwise encodes it twice: on a Moto G15 that doubled the time).
function(mv_patch_whisper src_dir out_file)
    file(READ ${src_dir}/whisper.cpp WCODE)
    set(A1 "    // auto-detect language if not specified\n")
    set(A2 "        state->lang_id = lang_id;\n        params.language = whisper_lang_str(lang_id);\n")
    set(A3 "#include <vector>\n")
    set(A4 "        // encode audio features starting at offset seek\n        if (!whisper_encode_internal(*ctx, *state, seek, params.n_threads, params.abort_callback, params.abort_callback_user_data)) {\n")
    foreach (anchor A1 A2 A3 A4)
        string(FIND "${WCODE}" "${${anchor}}" _pos)
        if (_pos EQUAL -1)
            message(FATAL_ERROR "whisper.cpp changed: patch anchor ${anchor} not found; update CMakeLists.txt")
        endif()
    endforeach()
    string(REPLACE "${A3}" "${A3}static thread_local std::vector<float> myvoice_last_lang_probs; // myvoice\nstatic thread_local bool myvoice_encoded_at_0 = false; // myvoice\n" WCODE "${WCODE}")
    string(REPLACE "${A1}" "    // myvoice: the shortened audio window applies to language detection as well\n    if (params.audio_ctx > 0 && params.audio_ctx <= whisper_n_audio_ctx(ctx)) {\n        state->exp_n_audio_ctx = params.audio_ctx;\n    }\n${A1}" WCODE "${WCODE}")
    string(REPLACE "${A2}" "${A2}        myvoice_last_lang_probs = probs; // myvoice\n        myvoice_encoded_at_0 = !params.detect_language; // myvoice: the encoder just ran on window 0\n" WCODE "${WCODE}")
    string(REPLACE "${A4}" "        // encode audio features starting at offset seek (myvoice: unless language detection just\n        // encoded this same window: whisper.cpp would otherwise run the encoder twice per sentence)\n        const bool myvoice_reuse = myvoice_encoded_at_0 && seek == 0;\n        myvoice_encoded_at_0 = false;\n        if (!myvoice_reuse && !whisper_encode_internal(*ctx, *state, seek, params.n_threads, params.abort_callback, params.abort_callback_user_data)) {\n" WCODE "${WCODE}")
    string(APPEND WCODE "\n// myvoice: detection probabilities of the last whisper_full() call on this thread\nextern \"C\" WHISPER_API const float * whisper_myvoice_lang_probs(int * n) {\n    *n = (int) myvoice_last_lang_probs.size();\n    return myvoice_last_lang_probs.data();\n}\n")
    file(WRITE ${out_file} "${WCODE}")
endfunction()
