# Builds the app's native code (src/main/cpp) for this laptop, CPU only, with a small command-line
# driver, so whisper.cpp / llama.cpp / espeak-ng can be checked against the Python tool without a
# phone. Used by ../check_native.py.
cmake_minimum_required(VERSION 3.22)
project(mv_host_test C CXX)
set(CMAKE_CXX_STANDARD 17)

set(TP ${CMAKE_CURRENT_LIST_DIR}/../../third_party)
set(CPP ${CMAKE_CURRENT_LIST_DIR}/../../myvoice/src/main/cpp)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# native CPU code on the laptop; -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=... when cross-compiling for the phone
set(GGML_NATIVE ON CACHE BOOL "")
foreach (opt LLAMA_BUILD_COMMON LLAMA_BUILD_TESTS LLAMA_BUILD_TOOLS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_SERVER LLAMA_BUILD_APP LLAMA_OPENSSL LLAMA_CURL)
    set(${opt} OFF CACHE BOOL "" FORCE)
endforeach()
add_subdirectory(${TP}/llama.cpp llama.cpp)

include(${CPP}/whisper_patch.cmake)
mv_patch_whisper(${TP}/whisper.cpp/src ${CMAKE_BINARY_DIR}/whisper_patched.cpp)
add_library(whisper STATIC ${CMAKE_BINARY_DIR}/whisper_patched.cpp ${TP}/whisper.cpp/src/aneforge/whisper-aneforge.cpp)
target_include_directories(whisper PUBLIC ${TP}/whisper.cpp/src ${TP}/whisper.cpp/include)
target_compile_definitions(whisper PRIVATE WHISPER_VERSION="1.9.5-myvoice")
target_link_libraries(whisper PUBLIC ggml)

foreach (opt USE_MBROLA USE_LIBSONIC USE_LIBPCAUDIO USE_ASYNC USE_KLATT USE_SPEECHPLAYER ESPEAK_BUILD_MANPAGES ESPEAK_COMPAT)
    set(${opt} OFF CACHE BOOL "" FORCE)
endforeach()
add_subdirectory(${TP}/espeak-ng espeak-ng EXCLUDE_FROM_ALL)

add_executable(host_test host_test.cpp ${CPP}/mv_backends.cpp ${CPP}/mv_whisper.cpp ${CPP}/mv_llama.cpp
               ${CPP}/mv_espeak.cpp ${CPP}/mv_devices.cpp)
target_include_directories(host_test PRIVATE ${CPP} ${TP}/OpenCL-Headers ${TP}/Vulkan-Headers/include)
target_link_libraries(host_test PRIVATE whisper llama ggml espeak-ng)
