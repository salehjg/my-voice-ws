package org.myvoice.audio;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Read and write simple WAV files (16-bit PCM or 32-bit float, any channel count -> mono). */
public final class Wav {
    private Wav() {
    }

    public static AudioClip read(File f) throws IOException {
        try (InputStream in = new FileInputStream(f)) {
            return read(in);
        }
    }

    public static AudioClip read(InputStream in) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[1 << 16];
        int n;
        while ((n = in.read(buf)) > 0) {
            bos.write(buf, 0, n);
        }
        ByteBuffer b = ByteBuffer.wrap(bos.toByteArray()).order(ByteOrder.LITTLE_ENDIAN);
        if (b.remaining() < 12 || b.getInt(0) != 0x46464952 || b.getInt(8) != 0x45564157) {   // "RIFF" ... "WAVE"
            throw new IOException("not a WAV file");
        }
        int pos = 12, format = 0, channels = 1, rate = 16000, bits = 16;
        while (pos + 8 <= b.limit()) {
            int id = b.getInt(pos), size = b.getInt(pos + 4);
            int body = pos + 8;
            if (id == 0x20746d66) {   // "fmt "
                format = b.getShort(body) & 0xffff;
                channels = b.getShort(body + 2);
                rate = b.getInt(body + 4);
                bits = b.getShort(body + 14);
            } else if (id == 0x61746164) {   // "data"
                size = Math.min(size < 0 ? Integer.MAX_VALUE : size, b.limit() - body);
                int bytesPer = bits / 8;
                int frames = size / (bytesPer * channels);
                float[] out = new float[frames];
                for (int i = 0; i < frames; i++) {
                    double sum = 0;
                    for (int ch = 0; ch < channels; ch++) {
                        int at = body + (i * channels + ch) * bytesPer;
                        if (format == 3 && bits == 32) {
                            sum += b.getFloat(at);
                        } else if (bits == 16) {
                            sum += b.getShort(at) / 32768.0;
                        } else if (bits == 32) {
                            sum += b.getInt(at) / 2147483648.0;
                        } else {
                            throw new IOException("unsupported WAV sample size: " + bits);
                        }
                    }
                    out[i] = (float) (sum / channels);
                }
                return new AudioClip(out, rate);
            }
            pos = body + size + (size & 1);
        }
        throw new IOException("WAV file without audio data");
    }

    public static void write(File f, AudioClip clip) throws IOException {
        try (OutputStream out = new FileOutputStream(f)) {
            int n = clip.samples().length;
            ByteBuffer b = ByteBuffer.allocate(44 + 2 * n).order(ByteOrder.LITTLE_ENDIAN);
            b.putInt(0x46464952).putInt(36 + 2 * n).putInt(0x45564157);
            b.putInt(0x20746d66).putInt(16).putShort((short) 1).putShort((short) 1)
                    .putInt(clip.sampleRate()).putInt(clip.sampleRate() * 2).putShort((short) 2).putShort((short) 16);
            b.putInt(0x61746164).putInt(2 * n);
            for (float v : clip.samples()) {
                b.putShort((short) Math.round(Math.max(-1f, Math.min(1f, v)) * 32767));
            }
            out.write(b.array());
        }
    }
}
