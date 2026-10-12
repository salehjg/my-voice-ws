package org.myvoice.core;

import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;

/** Thread-safe {@link Commands}: the UI and the notification put, the conversation thread takes. */
public final class CommandQueue implements Commands {
    private final LinkedBlockingDeque<Command> queue = new LinkedBlockingDeque<>();

    public void put(Command c) {
        queue.offerLast(c);
    }

    @Override
    public Command next(long timeoutMs) throws InterruptedException {
        if (timeoutMs < 0) {
            return queue.takeFirst();
        }
        if (timeoutMs == 0) {
            return queue.pollFirst();
        }
        return queue.pollFirst(timeoutMs, TimeUnit.MILLISECONDS);
    }

    @Override
    public void pushFront(Command c) {
        queue.offerFirst(c);
    }

    @Override
    public void clear() {
        queue.clear();
    }
}
