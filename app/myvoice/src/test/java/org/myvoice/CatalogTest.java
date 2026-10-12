package org.myvoice;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.FileInputStream;
import java.util.HashSet;
import java.util.Set;

import org.junit.Test;
import org.myvoice.models.Estimator;
import org.myvoice.models.ModelCatalog;
import org.myvoice.models.ModelCatalog.FileSpec;
import org.myvoice.models.ModelCatalog.Model;

/** assets/models.json is complete and consistent. */
public class CatalogTest {
    static ModelCatalog load() throws Exception {
        try (FileInputStream in = new FileInputStream("src/main/assets/models.json")) {
            return new ModelCatalog(in);
        }
    }

    @Test
    public void everyModelIsDownloadable() throws Exception {
        ModelCatalog cat = load();
        Set<String> paths = new HashSet<>();
        for (Model m : cat.models) {
            assertFalse(m.id(), m.files().isEmpty());
            for (FileSpec f : m.files()) {
                assertTrue(f.url(), f.url().startsWith("https://"));
                assertTrue(f.path(), f.size() > 0);
                assertTrue("duplicate path " + f.path(), paths.add(f.path()));
                boolean big = f.path().endsWith(".onnx") || f.path().endsWith(".bin") || f.path().endsWith(".gguf");
                if (big) {
                    assertEquals(f.path() + " needs a sha256", 64, f.sha256().length());
                }
            }
        }
    }

    @Test
    public void presetsAreComplete() throws Exception {
        ModelCatalog cat = load();
        assertNotNull(cat.preset(cat.defaultPreset));
        assertEquals("laptop", cat.defaultPreset);
        for (ModelCatalog.Preset p : cat.presets) {
            Set<String> parts = new HashSet<>();
            for (String id : p.models()) {
                Model m = cat.get(id);
                assertNotNull(p.id() + ": " + id, m);
                parts.add(m.component().equals("voice") ? "voice_" + m.lang() : m.component());
            }
            assertEquals(p.id(), Set.of("asr", "mt", "voice_it", "voice_en"), parts);
        }
        // the laptop preset is exactly the laptop's translation model file
        assertTrue(cat.get("translategemma-4b-q6_k").files().get(0).path().endsWith("translategemma-4b-it.Q6_K.gguf"));
    }

    @Test
    public void estimatesComeFromTheReferencePhone() throws Exception {
        ModelCatalog cat = load();
        assertEquals("moto g15", cat.referenceDevice);
        Estimator g15 = new Estimator(cat.referenceGflops, cat.referenceGbps);
        // on the reference phone: the measured numbers; scaled ones agree when the benchmark matches
        assertEquals(1.5, g15.seconds(cat.get("whisper-base-q8_0"), cat, true), 1e-9);
        assertEquals(39.0, g15.seconds(cat.get("translategemma-4b-q6_k"), cat, false), 1e-6);
        Estimator twiceAsFast = new Estimator(2 * cat.referenceGflops, 2 * cat.referenceGbps);
        assertEquals(0.75, twiceAsFast.seconds(cat.get("whisper-base-q8_0"), cat, false), 1e-9);
        assertTrue(g15.seconds(cat.get("whisper-base-q8_0"), cat, true) < g15.seconds(cat.get("whisper-small-q8_0"), cat, true));
        assertTrue(g15.seconds(cat.get("whisper-small-q8_0"), cat, true) < g15.seconds(cat.get("whisper-large-v3-turbo-q5_0"), cat, true));
        assertTrue(g15.seconds(cat.get("opus-mt-int8"), cat, true) < g15.seconds(cat.get("gemma-3-1b-q4_0"), cat, true));
        assertTrue(g15.seconds(cat.get("gemma-3-1b-q4_0"), cat, true) < g15.seconds(cat.get("translategemma-4b-q6_k"), cat, true));
        assertEquals("too big for this phone's memory", Estimator.ramFit(4L << 30, 3600));
        assertEquals("fits in memory", Estimator.ramFit(8L << 30, 400));
    }
}
