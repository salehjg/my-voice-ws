package org.myvoice;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Map;

import org.junit.Test;
import org.myvoice.asr.HallucinationFilter;
import org.myvoice.asr.LanguageChooser;
import org.myvoice.lang.Languages;
import org.myvoice.mt.OutputCleaner;
import org.myvoice.mt.Prompts;

/** Ports of tests/test_config_langs.py and tests/test_translate.py. */
public class TextLogicTest {
    private static final List<String> PAIR = List.of("it", "en");

    @Test
    public void directions() {
        assertNull(Languages.parseDirection("auto", PAIR));
        for (String m : new String[] {"it-en", "it->en", "it2en", "IT-EN"}) {
            assertArrayEquals(new String[] {"it", "en"}, Languages.parseDirection(m, PAIR));
        }
        assertArrayEquals(new String[] {"en", "it"}, Languages.parseDirection("en-it", PAIR));
        assertEquals("en", Languages.other("it", PAIR));
        assertThrows(IllegalArgumentException.class, () -> Languages.parseDirection("it-fr", PAIR));
    }

    @Test
    public void guessTextLanguage() {
        Object[][] cases = {
                {"Ciao, come stai? Vorrei un caffè per favore.", "it"},
                {"Dov'è la stazione dei treni?", "it"},
                {"Where is the train station, please?", "en"},
                {"I would like a coffee and a croissant.", "en"},
                {"Perché non andiamo al mare domani?", "it"},
        };
        for (Object[] c : cases) {
            assertEquals((String) c[0], c[1], Languages.guessTextLanguage((String) c[0], PAIR));
        }
    }

    @Test
    public void translateGemmaPromptMatchesOfficialWording() {
        String s = Prompts.translateGemmaInstruction("Ciao!", Languages.get("it"), Languages.get("en"));
        assertTrue(s.startsWith("You are a professional Italian (it) to English (en) translator."));
        assertTrue(s.contains("cultural sensitivities.\nProduce only the English translation, without any additional explanations or commentary."));
        assertTrue(s.endsWith("Please translate the following Italian text into English:\n\n\nCiao!"));
        String p = Prompts.translateGemma("Ciao!", Languages.get("it"), Languages.get("en"));
        assertTrue(p.startsWith("<start_of_turn>user\nYou are"));
        assertTrue(p.endsWith("Ciao!<end_of_turn>\n<start_of_turn>model\n"));
    }

    @Test
    public void chatPromptIncludesContext() {
        String sys = Prompts.chatSystem(Languages.get("it"), Languages.get("en"),
                List.of(new Prompts.Exchange("en", "Can you help me?", "it", "Mi può aiutare?")));
        assertTrue(sys.contains("Italian speech"));
        assertTrue(sys.endsWith("- English: Can you help me?\n  Italian: Mi può aiutare?"));
        String g = Prompts.gemmaChat("Sì, certo.", Languages.get("it"), Languages.get("en"), List.of());
        assertTrue(g.startsWith("<start_of_turn>user\nYou are a professional simultaneous interpreter."));
        assertTrue(g.endsWith("\n\nSì, certo.<end_of_turn>\n<start_of_turn>model\n"));
    }

    @Test
    public void cleanOutput() {
        String[][] cases = {
                {"Good morning, how are you?", "Buongiorno", "Good morning, how are you?"},
                {"  \"Good morning.\"  ", "Buongiorno.", "Good morning."},
                {"“Buongiorno.”", "Good morning.", "Buongiorno."},
                {"\"Hi,\" she said.", "\"Ciao,\" disse.", "\"Hi,\" she said."},
                {"<think>\nthe user wants...\n</think>\n\nHello!", "Ciao!", "Hello!"},
                {"Translation: Hello!", "Ciao!", "Hello!"},
                {"Hello there!\n\nNote: 'Ciao' can mean hi or bye.", "Ciao!", "Hello there!"},
                {"Ciao!<end_of_turn>junk", "Hi!", "Ciao!"},
                {"Line one.\nLine two.", "Uno. Due.", "Line one. Line two."},
                {"<think>never finished", "x", ""},
        };
        for (String[] c : cases) {
            assertEquals(c[0], c[2], OutputCleaner.clean(c[0], c[1]));
        }
    }

    @Test
    public void maxTokensScalesWithInput() {
        assertEquals(64 + 12, Prompts.maxTokensFor("uno due tre", 100));
        assertEquals(100, Prompts.maxTokensFor("parola ".repeat(200), 100));
    }

    @Test
    public void hallucinationFilter() {
        assertTrue(HallucinationFilter.isHallucination("Sottotitoli creati dalla comunità Amara.org"));
        assertTrue(HallucinationFilter.isHallucination("Sottotitoli a cura di QTSS"));
        assertTrue(HallucinationFilter.isHallucination("Thank you for watching!"));
        assertTrue(HallucinationFilter.isHallucination("Grazie per la visione."));
        assertTrue(HallucinationFilter.isHallucination("..."));
        assertFalse(HallucinationFilter.isHallucination("Grazie mille, a domani!"));
        assertFalse(HallucinationFilter.isHallucination("Thank you for coming."));
    }

    @Test
    public void chooseLanguageRestrictedToPair() {
        LanguageChooser.Choice c = LanguageChooser.choose(
                Map.of("es", 0.55f, "it", 0.35f, "en", 0.05f, "pt", 0.05f), "es", 0.55f, PAIR);
        assertEquals("it", c.language());
        assertEquals(0.35 / 0.40, c.confidence(), 1e-4);
        LanguageChooser.Choice d = LanguageChooser.choose(null, "en", 0.9f, PAIR);
        assertEquals("en", d.language());
        assertEquals(1.0, d.confidence(), 1e-6);
    }
}
