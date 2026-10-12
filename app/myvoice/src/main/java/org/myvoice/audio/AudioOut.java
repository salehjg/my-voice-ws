package org.myvoice.audio;

/** A speaker: plays one clip at a time in the background. */
public interface AudioOut {
    void start(AudioClip clip);

    boolean isPlaying();

    void stop();
}
