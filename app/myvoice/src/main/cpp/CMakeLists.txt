# Native part of the app: ggml (CPU variants + Vulkan + OpenCL as loadable modules), llama.cpp,
# whisper.cpp, espeak-ng, and the JNI bridge (libmyvoice_jni.so).

cmake_minimum_required(VERSION 3.22)
project(myvoice_native C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT MYVOICE_THIRD_PARTY)
    set(MYVOICE_THIRD_PARTY ${CMAKE_CURRENT_LIST_DIR}/../../../../third_party)
endif()
set(TP ${MYVOICE_THIRD_PARTY})
foreach (dep llama.cpp whisper.cpp espeak-ng Vulkan-Headers SPIRV-Headers OpenCL-Headers OpenCL-ICD-Loader)
    if (NOT EXISTS ${TP}/${dep}/CMakeLists.txt AND NOT EXISTS ${TP}/${dep}/include)
        message(FATAL_ERROR "${TP}/${dep} is missing: run `git submodule update --init` in the repository")
    endif()
endforeach()

option(MYVOICE_VULKAN "Build the ggml Vulkan backend" ON)
option(MYVOICE_OPENCL "Build the ggml OpenCL backend" ON)

# ------------------------------------------------------------------ OpenCL (link-time only)
# ggml-opencl links against libOpenCL.so; the Khronos ICD loader provides it at build time and is
# left out of the APK, so the phone's own driver is used (<uses-native-library> in the manifest).
if (MYVOICE_OPENCL)
    set(OPENCL_ICD_LOADER_HEADERS_DIR ${TP}/OpenCL-Headers CACHE PATH "" FORCE)
    set(OPENCL_ICD_LOADER_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    add_subdirectory(${TP}/OpenCL-ICD-Loader opencl-icd-loader EXCLUDE_FROM_ALL)
    set(OpenCL_INCLUDE_DIR ${TP}/OpenCL-Headers CACHE PATH "" FORCE)
    set(OpenCL_LIBRARY OpenCL CACHE STRING "" FORCE)
endif()

# ------------------------------------------------------------------ Vulkan
if (MYVOICE_VULKAN)
    # vulkan.hpp comes from Vulkan-Headers; libvulkan.so from the NDK; glslc from the NDK's shader tools
    set(Vulkan_INCLUDE_DIR ${TP}/Vulkan-Headers/include CACHE PATH "" FORCE)
    set(Vulkan_LIBRARY ${CMAKE_SYSROOT}/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/${ANDROID_PLATFORM_LEVEL}/libvulkan.so CACHE FILEPATH "" FORCE)
    if (NOT EXISTS ${Vulkan_LIBRARY})
        file(GLOB _vk ${CMAKE_SYSROOT}/usr/lib/${ANDROID_TOOLCHAIN_NAME}/${ANDROID_PLATFORM_LEVEL}/libvulkan.so)
        set(Vulkan_LIBRARY ${_vk} CACHE FILEPATH "" FORCE)
    endif()
    set(Vulkan_GLSLC_EXECUTABLE ${ANDROID_NDK}/shader-tools/linux-x86_64/glslc CACHE FILEPATH "" FORCE)
    # SPIRV-Headers is header-only; give find_package(SPIRV-Headers CONFIG) a minimal config
    file(WRITE ${CMAKE_BINARY_DIR}/spirv-headers/SPIRV-HeadersConfig.cmake
        "if (NOT TARGET SPIRV-Headers::SPIRV-Headers)\n"
        "    add_library(SPIRV-Headers::SPIRV-Headers INTERFACE IMPORTED)\n"
        "    set_target_properties(SPIRV-Headers::SPIRV-Headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES \"${TP}/SPIRV-Headers/include\")\n"
        "endif()\n")
    set(SPIRV-Headers_DIR ${CMAKE_BINARY_DIR}/spirv-headers CACHE PATH "" FORCE)
endif()

# ------------------------------------------------------------------ ggml + llama.cpp
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)        # needed for loadable backends
set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)     # armv8.0 ... armv9.2: the best one is picked at run time
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ${MYVOICE_VULKAN} CACHE BOOL "" FORCE)
set(GGML_OPENCL ${MYVOICE_OPENCL} CACHE BOOL "" FORCE)
set(GGML_OPENCL_EMBED_KERNELS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory(${TP}/llama.cpp llama.cpp)
if (TARGET ggml-vulkan)
    # ggml-vulkan finds SPIRV-Headers but expects its headers on a Vulkan SDK include path
    target_include_directories(ggml-vulkan PRIVATE ${TP}/SPIRV-Headers/include)
endif()

# ------------------------------------------------------------------ whisper.cpp (patched copy)
# Two small changes, made to a copy in the build folder (the submodule stays untouched):
#  1. the shortened audio window (audio_ctx) is used for language detection too, not only for
#     transcription, which matters on slow phones;
#  2. the language-detection probabilities of the last run are kept for the caller, so the app
#     can restrict detection to Italian/English without a second detection pass.
include(${CMAKE_CURRENT_LIST_DIR}/whisper_patch.cmake)
mv_patch_whisper(${TP}/whisper.cpp/src ${CMAKE_BINARY_DIR}/whisper_patched.cpp)
set(WSRC ${TP}/whisper.cpp/src)

add_library(whisper SHARED ${CMAKE_BINARY_DIR}/whisper_patched.cpp ${WSRC}/aneforge/whisper-aneforge.cpp)
target_include_directories(whisper PUBLIC ${WSRC} ${TP}/whisper.cpp/include)
target_compile_definitions(whisper PRIVATE WHISPER_SHARED WHISPER_BUILD WHISPER_VERSION="1.9.5-myvoice")
target_link_libraries(whisper PUBLIC ggml)

# ------------------------------------------------------------------ espeak-ng (library only)
set(USE_MBROLA OFF CACHE BOOL "" FORCE)
set(USE_LIBSONIC OFF CACHE BOOL "" FORCE)
set(USE_LIBPCAUDIO OFF CACHE BOOL "" FORCE)
set(USE_ASYNC OFF CACHE BOOL "" FORCE)
set(USE_KLATT OFF CACHE BOOL "" FORCE)
set(USE_SPEECHPLAYER OFF CACHE BOOL "" FORCE)
set(ESPEAK_BUILD_MANPAGES OFF CACHE BOOL "" FORCE)
set(ESPEAK_COMPAT OFF CACHE BOOL "" FORCE)
add_subdirectory(${TP}/espeak-ng espeak-ng EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------ the app's library
add_library(myvoice_jni SHARED
    jni_bridge.cpp
    mv_backends.cpp
    mv_whisper.cpp
    mv_llama.cpp
    mv_espeak.cpp
    mv_devices.cpp)
target_include_directories(myvoice_jni PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${TP}/OpenCL-Headers ${TP}/Vulkan-Headers/include)
target_link_libraries(myvoice_jni PRIVATE whisper llama ggml ggml-base espeak-ng log dl)

# ggml's backend modules are loaded with dlopen(), so nothing links them: build them explicitly.
get_property(_ggml_src_targets DIRECTORY ${TP}/llama.cpp/ggml/src PROPERTY BUILDSYSTEM_TARGETS)
foreach (t ${_ggml_src_targets} ggml-vulkan ggml-opencl)
    if (TARGET ${t} AND t MATCHES "^ggml-(cpu.*|vulkan|opencl)$")
        add_dependencies(myvoice_jni ${t})
    endif()
endforeach()
