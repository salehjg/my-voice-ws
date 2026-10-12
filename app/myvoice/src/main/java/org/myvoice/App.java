package org.myvoice;

import android.app.Application;

/** Creates the app's own folders in Android/data at start, so files copied there with adb are readable. */
public final class App extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        // The app can't read folders that `adb push` creates itself in Android/data, but it can read
        // files pushed into folders it made: test_audio/ (debug input) and sessions/ (transcripts).
        getExternalFilesDir("test_audio");
        getExternalFilesDir("sessions");
    }
}
