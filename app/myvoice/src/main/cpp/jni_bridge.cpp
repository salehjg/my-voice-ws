// JNI wrappers for org.myvoice.nativebridge.NativeLib (see mv_native.h for what each call does).

#include "mv_native.h"

#include <jni.h>

#include <string>
#include <vector>

namespace {

// Java strings <-> real UTF-8 (JNI's own "modified UTF-8" mangles emoji)
std::string str(JNIEnv * env, jstring s) {
    if (!s) {
        return "";
    }
    jclass cls = env->FindClass("java/lang/String");
    jmethodID getBytes = env->GetMethodID(cls, "getBytes", "(Ljava/lang/String;)[B");
    jstring utf8 = env->NewStringUTF("UTF-8");
    auto bytes = (jbyteArray) env->CallObjectMethod(s, getBytes, utf8);
    jsize n = env->GetArrayLength(bytes);
    std::string out(n, '\0');
    env->GetByteArrayRegion(bytes, 0, n, (jbyte *) out.data());
    env->DeleteLocalRef(bytes);
    env->DeleteLocalRef(utf8);
    env->DeleteLocalRef(cls);
    return out;
}

jstring jstr(JNIEnv * env, const std::string & s) {
    jclass cls = env->FindClass("java/lang/String");
    jmethodID ctor = env->GetMethodID(cls, "<init>", "([BLjava/lang/String;)V");
    jbyteArray bytes = env->NewByteArray((jsize) s.size());
    env->SetByteArrayRegion(bytes, 0, (jsize) s.size(), (const jbyte *) s.data());
    jstring utf8 = env->NewStringUTF("UTF-8");
    auto out = (jstring) env->NewObject(cls, ctor, bytes, utf8);
    env->DeleteLocalRef(bytes);
    env->DeleteLocalRef(utf8);
    env->DeleteLocalRef(cls);
    return out;
}

jobjectArray jarr(JNIEnv * env, const std::vector<std::string> & v) {
    jclass cls = env->FindClass("java/lang/String");
    jobjectArray a = env->NewObjectArray((jsize) v.size(), cls, nullptr);
    for (size_t i = 0; i < v.size(); i++) {
        jstring s = jstr(env, v[i]);
        env->SetObjectArrayElement(a, (jsize) i, s);
        env->DeleteLocalRef(s);
    }
    env->DeleteLocalRef(cls);
    return a;
}

}  // namespace

#define FN(name) JNICALL Java_org_myvoice_nativebridge_NativeLib_##name

extern "C" {

JNIEXPORT jobjectArray FN(ggmlLoadBackends)(JNIEnv * env, jclass, jstring dir, jstring gpu) {
    return jarr(env, mv_load_backends(str(env, dir), str(env, gpu)));
}

JNIEXPORT jstring FN(ggmlCpuFeatures)(JNIEnv * env, jclass) {
    return jstr(env, mv_cpu_features());
}

JNIEXPORT jlong FN(whisperInit)(JNIEnv * env, jclass, jstring model, jstring device, jboolean flash) {
    return (jlong) mv_whisper_init(str(env, model), str(env, device), flash);
}

JNIEXPORT jstring FN(whisperTranscribe)(JNIEnv * env, jclass, jlong ctx, jfloatArray audio, jstring language,
                                        jint threads, jstring prompt, jint audio_ctx, jfloatArray probs) {
    jsize n = env->GetArrayLength(audio);
    std::vector<float> pcm(n);
    env->GetFloatArrayRegion(audio, 0, n, pcm.data());
    std::vector<float> p;
    if (probs) {
        p.resize(env->GetArrayLength(probs), 0.0f);
    }
    bool ok = false;
    std::string out = mv_whisper_transcribe((void *) ctx, pcm.data(), n, str(env, language), threads, str(env, prompt),
                                            audio_ctx, probs ? p.data() : nullptr, (int) p.size(), ok);
    if (!ok) {
        return nullptr;
    }
    if (probs) {
        env->SetFloatArrayRegion(probs, 0, (jsize) p.size(), p.data());
    }
    return jstr(env, out);
}

JNIEXPORT jint FN(whisperLangId)(JNIEnv * env, jclass, jstring code) {
    return mv_whisper_lang_id(str(env, code));
}

JNIEXPORT jint FN(whisperLangCount)(JNIEnv *, jclass) {
    return mv_whisper_lang_count();
}

JNIEXPORT void FN(whisperFree)(JNIEnv *, jclass, jlong ctx) {
    mv_whisper_free((void *) ctx);
}

JNIEXPORT jlong FN(llamaInit)(JNIEnv * env, jclass, jstring model, jstring device, jint gpu_layers, jint n_ctx, jint threads) {
    return (jlong) mv_llama_init(str(env, model), str(env, device), gpu_layers, n_ctx, threads);
}

JNIEXPORT jstring FN(llamaComplete)(JNIEnv * env, jclass, jlong ctx, jstring prompt, jint max_tokens, jobjectArray stops) {
    std::vector<std::string> s;
    for (jsize i = 0; stops && i < env->GetArrayLength(stops); i++) {
        auto e = (jstring) env->GetObjectArrayElement(stops, i);
        s.push_back(str(env, e));
        env->DeleteLocalRef(e);
    }
    bool ok = false;
    std::string out = mv_llama_complete((void *) ctx, str(env, prompt), max_tokens, s, ok);
    return ok ? jstr(env, out) : nullptr;
}

JNIEXPORT void FN(llamaFree)(JNIEnv *, jclass, jlong ctx) {
    mv_llama_free((void *) ctx);
}

JNIEXPORT jint FN(espeakInit)(JNIEnv * env, jclass, jstring dir) {
    return mv_espeak_init(str(env, dir));
}

JNIEXPORT jint FN(espeakSetVoice)(JNIEnv * env, jclass, jstring voice) {
    return mv_espeak_set_voice(str(env, voice));
}

JNIEXPORT jobjectArray FN(espeakPhonemize)(JNIEnv * env, jclass, jstring text) {
    return jarr(env, mv_espeak_phonemize(str(env, text)));
}

JNIEXPORT jobjectArray FN(vulkanDevices)(JNIEnv * env, jclass) {
    return jarr(env, mv_vulkan_devices());
}

JNIEXPORT jobjectArray FN(openclDevices)(JNIEnv * env, jclass) {
    return jarr(env, mv_opencl_devices());
}

JNIEXPORT jobjectArray FN(nnapiDevices)(JNIEnv * env, jclass) {
    return jarr(env, mv_nnapi_devices());
}

}  // extern "C"
