package org.myvoice.audio;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;

import android.content.Context;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;

/** Decodes an audio file (WhatsApp voice note, mp3, m4a, wav, ...) to 16 kHz mono with the phone's codecs. */
public final class AudioDecoder {
    private AudioDecoder() {
    }

    public static AudioClip decode16k(Context context, Uri uri) throws Exception {
        MediaExtractor ex = new MediaExtractor();
        ex.setDataSource(context, uri, null);
        int track = -1;
        MediaFormat fmt = null;
        for (int i = 0; i < ex.getTrackCount(); i++) {
            MediaFormat f = ex.getTrackFormat(i);
            String mime = f.getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith("audio/")) {
                track = i;
                fmt = f;
                break;
            }
        }
        if (track < 0) {
            ex.release();
            throw new IllegalArgumentException("no audio in this file");
        }
        ex.selectTrack(track);
        MediaCodec codec = MediaCodec.createDecoderByType(fmt.getString(MediaFormat.KEY_MIME));
        codec.configure(fmt, null, null, 0);
        codec.start();
        int rate = fmt.getInteger(MediaFormat.KEY_SAMPLE_RATE);
        int channels = fmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
        ByteArrayOutputStream pcm = new ByteArrayOutputStream();
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        boolean inputDone = false;
        try {
            while (true) {
                if (!inputDone) {
                    int in = codec.dequeueInputBuffer(10_000);
                    if (in >= 0) {
                        ByteBuffer buf = codec.getInputBuffer(in);
                        int n = ex.readSampleData(buf, 0);
                        if (n < 0) {
                            codec.queueInputBuffer(in, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                            inputDone = true;
                        } else {
                            codec.queueInputBuffer(in, 0, n, ex.getSampleTime(), 0);
                            ex.advance();
                        }
                    }
                }
                int out = codec.dequeueOutputBuffer(info, 10_000);
                if (out == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    MediaFormat of = codec.getOutputFormat();
                    rate = of.getInteger(MediaFormat.KEY_SAMPLE_RATE);
                    channels = of.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
                } else if (out >= 0) {
                    ByteBuffer ob = codec.getOutputBuffer(out);
                    byte[] chunk = new byte[info.size];
                    ob.position(info.offset);
                    ob.get(chunk);
                    pcm.write(chunk);
                    codec.releaseOutputBuffer(out, false);
                    if ((info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                        break;
                    }
                }
            }
        } finally {
            codec.stop();
            codec.release();
            ex.release();
        }
        ShortBuffer sb = ByteBuffer.wrap(pcm.toByteArray()).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer();
        int frames = sb.remaining() / channels;
        float[] mono = new float[frames];
        for (int i = 0; i < frames; i++) {
            float sum = 0;
            for (int c = 0; c < channels; c++) {
                sum += sb.get(i * channels + c) / 32768f;
            }
            mono[i] = sum / channels;
        }
        return new AudioClip(mono, rate).resampled(16000);
    }
}
