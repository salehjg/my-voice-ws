package org.myvoice.core;

/** The queue of user commands the conversation reads from. */
public interface Commands {
    /** The next command; null if none within {@code timeoutMs} (0 = just look, negative = wait). */
    Command next(long timeoutMs) throws InterruptedException;

    /** Put a command back so it is the next one read. */
    void pushFront(Command c);

    /** Forget commands that were queued (e.g. taps made while a sentence was playing). */
    void clear();
}
