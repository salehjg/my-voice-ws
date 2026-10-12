package org.myvoice.ui;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import android.view.LayoutInflater;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.RecyclerView;

import org.myvoice.R;
import org.myvoice.core.TurnResult;
import org.myvoice.databinding.ItemTurnBinding;
import org.myvoice.lang.Languages;

/** The conversation, one card per translated sentence (like the laptop's panels). */
final class TurnAdapter extends RecyclerView.Adapter<TurnAdapter.Holder> {
    private final List<TurnResult> turns = new ArrayList<>();
    private final String theirs;

    TurnAdapter(String theirLanguage) {
        theirs = theirLanguage;
    }

    void set(List<TurnResult> all) {
        turns.clear();
        turns.addAll(all);
        notifyDataSetChanged();
    }

    void add(TurnResult r) {
        turns.add(r);
        notifyItemInserted(turns.size() - 1);
    }

    static final class Holder extends RecyclerView.ViewHolder {
        final ItemTurnBinding b;

        Holder(ItemTurnBinding b) {
            super(b.getRoot());
            this.b = b;
        }
    }

    @NonNull
    @Override
    public Holder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        return new Holder(ItemTurnBinding.inflate(LayoutInflater.from(parent.getContext()), parent, false));
    }

    @Override
    public void onBindViewHolder(@NonNull Holder h, int i) {
        TurnResult r = turns.get(i);
        int color = ContextCompat.getColor(h.b.getRoot().getContext(), r.srcLang.equals(theirs) ? R.color.their : R.color.mine);
        String title = Languages.name(r.srcLang) + " → " + Languages.name(r.tgtLang);
        h.b.title.setText(r.who.isEmpty() ? title : r.who + " · " + title);
        h.b.title.setTextColor(color);
        h.b.source.setText(r.srcText);
        h.b.target.setText(r.tgtText);
        h.b.target.setTextColor(color);
        h.b.card.setStrokeColor(color);
        String footer = r.timingText();
        if (r.langProb < 0.8f) {   // detected and unsure: point at the fix
            footer = String.format(Locale.ROOT, "%s? %.0f%% sure — tap Wrong language · %s", Languages.name(r.srcLang),
                    r.langProb * 100, footer);
        }
        h.b.footer.setText(footer);
    }

    @Override
    public int getItemCount() {
        return turns.size();
    }
}
