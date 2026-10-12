package org.myvoice.audio;

import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;

/** Plays one clip at a time through the speaker (or headphones); {@link #stop()} cuts it off. */
public final class AudioTrackPlayer implements AudioOut {
    private AudioTrack track;
    private int frames;

    @Override
    public synchronized void start(AudioClip clip) {
        stop();
        if (clip.isEmpty()) {
            return;
        }
        float[] s = clip.samples();
        frames = s.length;
        track = new AudioTrack.Builder()
                .setAudioAttributes(new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                .setAudioFormat(new AudioFormat.Builder().setSampleRate(clip.sampleRate())
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).setEncoding(AudioFormat.ENCODING_PCM_FLOAT).build())
                .setTransferMode(AudioTrack.MODE_STATIC)
                .setBufferSizeInBytes(s.length * 4)
                .build();
        track.write(s, 0, s.length, AudioTrack.WRITE_BLOCKING);
        track.play();
    }

    @Override
    public synchronized boolean isPlaying() {
        if (track == null) {
            return false;
        }
        boolean playing = track.getPlayState() == AudioTrack.PLAYSTATE_PLAYING && track.getPlaybackHeadPosition() < frames;
        if (!playing) {
            release();
        }
        return playing;
    }

    @Override
    public synchronized void stop() {
        if (track != null) {
            try {
                track.stop();
            } catch (IllegalStateException ignored) {
            }
            release();
        }
    }

    private void release() {
        if (track != null) {
            track.release();
            track = null;
        }
    }
}
