import unittest

from app.router import route_segment
from app.schema import Segment
from app.speech_text import spoken_math


class SpokenMathTests(unittest.TestCase):
    """Real lines from the Minima-of-a-function lesson: the voice read them as 'dollar x dollar' and subtitles showed the markup."""

    def test_inline_math_loses_its_delimiters(self):
        self.assertEqual(spoken_math("The question states that for a real number $ x $, when it is added to its reciprocal"),
                         "The question states that for a real number x, when it is added to its reciprocal")
        self.assertEqual(spoken_math("Let us denote this sum as a function $ f(x) $."), "Let us denote this sum as a function f of x.")

    def test_fractions_powers_and_relations_become_words(self):
        self.assertEqual(spoken_math("Therefore, $ f(x) = x + \\frac{1}{x} $."), "Therefore, f of x equals x plus 1 over x.")
        text = spoken_math("$$ f'(x) = 1 - \\frac{1}{x^2} $$ Setting $ f'(x) = 0 $, we get")
        self.assertIn("f dash of x equals 1 minus 1 over x squared", text)
        self.assertIn("f dash of x equals 0, we get", text)
        text = spoken_math("$$ f''(1) = \\frac{2}{1^3} = 2 > 0 $$")
        self.assertIn("f double dash of 1 equals 2 over 1 cubed equals 2 is greater than 0", text)
        self.assertIn("x equals plus or minus 1", spoken_math("so $ x = \\pm 1 $"))
        self.assertIn("x squared equals 1 which implies x", spoken_math("$ x^2 = 1 \\Rightarrow x = 1 $"))

    def test_nothing_of_the_markup_survives(self):
        samples = ["$$ f''(-1) = \\frac{2}{(-1)^3} = \\frac{2}{-1} = -2 < 0 $$ Since it is negative", "at $ x = a $", "$ \\frac{1}{x} $ to $ x $",
                   "a cut-off formula $$ x 2 = ", "\\frac{1}{x^{2}} stray command"]
        for sample in samples:
            out = spoken_math(sample)
            for bad in ("$", "\\", "{", "}", "frac"):
                self.assertNotIn(bad, out, (sample, out))
        self.assertNotIn("( ", spoken_math("$ \\frac{2}{(-1)^3} $"))

    def test_plain_prose_is_untouched(self):
        for text in ["Anita wants to gift him a silver-plated keychain.", "The cost is 5 dollars; we add 2 and 3 - quickly."]:
            self.assertEqual(spoken_math(text), text)

    def test_non_english_keeps_symbols_not_english_words(self):
        out = spoken_math("అప్పుడు $ x \\pm 1 $ మరియు $ \\frac{a}{b} $", english=False)
        self.assertIn("±", out)
        self.assertIn("(a)/(b)", out)
        self.assertNotIn("plus or minus", out)

    def test_contract_subtitle_field_is_normalised_so_it_matches_the_voice(self):
        """The subtitle field (Java) kept the LaTeX and was shown instead of the narration, with rough timing."""
        from app.captions import _display_text
        from app.schema import Scene, Shot, Storyboard
        from app.storyboard_cleanup import normalize_storyboard
        raw = "The first derivative is given by: $$ f'(x) = 1 - \\frac{1}{x^2} $$ Setting $ f'(x) = 0 $, we get:"
        seg = Segment(segment_number=1, narration=raw, subtitle=raw, shot=Shot(template="photo", heading="x"))
        board = Storyboard(title="Minimizing f(x) = x + 1/x Using Calculus",
                           scenes=[Scene(scene_number=1, title="Derivative of $ f(x) $", narration=raw, segments=[seg])])
        normalize_storyboard(board)
        for text in (seg.narration, seg.subtitle, board.scenes[0].title, board.title):
            self.assertNotIn("$", text)
            self.assertNotIn("\\", text)
        self.assertEqual(seg.subtitle, seg.narration)
        self.assertEqual(_display_text(seg), seg.narration)       # so the exact voice word timings are used for the subtitle
        self.assertIn("f dash of x equals 1 minus 1 over x squared", seg.subtitle)

    def test_subtitle_cues_end_at_sentences_when_the_voice_timings_have_no_punctuation(self):
        from app.captions import build_cues
        from app.config import AppConfig
        from app.schema import Caption
        text = ("Setting f dash of x equals 0, we get: 1 minus 1 over x squared equals 0. This simplifies to 1 over x squared equals 1. "
                "From this it follows that x squared equals 1.")
        words = text.split()
        stripped = [w.strip(".,:") for w in words]                  # what the speech engine reports
        seg = Segment(segment_number=1, narration=text, start=0.0, end=len(words) * 0.4, speech_duration=len(words) * 0.4,
                      captions=[Caption(text=w, start=i * 0.4, end=(i + 1) * 0.4) for i, w in enumerate(stripped)])
        cfg = AppConfig()
        cues = build_cues(seg, cfg.production.subtitle, (1920, 1080))
        self.assertGreaterEqual(len(cues), 3)                       # one cue per sentence at least, not one block
        self.assertTrue(cues[0].text.endswith("we get:"))           # punctuation is back
        self.assertTrue(all(cue.text[-1] in ".:" for cue in cues[:-1]))
        # mismatched word counts fall back to the engine's words
        broken = Segment(segment_number=2, narration=text, start=0.0, end=4.0, speech_duration=4.0,
                         captions=[Caption(text="only", start=0.0, end=1.0), Caption(text="three", start=1.0, end=2.0)])
        self.assertEqual([c.text for c in build_cues(broken, cfg.production.subtitle, (1920, 1080))][0][:4], "only")

    def test_a_sentence_with_a_colon_is_not_split_into_term_and_body(self):
        """Real card: 'Let's now work through question number 176, which asks: Which of the following ...' lost its body to one line."""
        from PIL import Image
        from app.charts import cards_frame
        from app.config import AppConfig
        config = AppConfig()
        long_sentence = "Let's now work through question number 176, which asks: Which of the following is a commercial blood cholesterol-lowering agent?"
        sentence_card = cards_frame((1920, 1080), config, "Question", [long_sentence, "Cholesterol-lowering agent"], 1, 1.0, False)
        term_card = cards_frame((1920, 1080), config, "Question", ["Statin: lowers blood cholesterol levels", "Another"], 1, 1.0, False)
        for image in (sentence_card, term_card):
            self.assertEqual(image.size, (1920, 1080))
        # the sentence card draws far more text pixels (all five lines available) than the squeezed term/body layout would
        dark = lambda im: sum(1 for p in im.crop((96, 330, 900, 700)).convert("L").getdata() if p < 110)
        self.assertGreater(dark(sentence_card), 2500)

    def test_fragment_formula_lines_are_dropped(self):
        from app.schema import Scene, Shot, Storyboard
        from app.storyboard_cleanup import normalize_storyboard
        seg = Segment(segment_number=1, narration="Then.", formula_lines=["L' = 1.1L_0", "=", "  ", "= 1.1L_0"], shot=Shot(template="photo", heading="x"))
        board = Storyboard(title="t", scenes=[Scene(scene_number=1, title="s", narration="Then.", segments=[seg])])
        normalize_storyboard(board)
        self.assertEqual(seg.formula_lines, ["L' = 1.1L_0", "= 1.1L_0"])

    def test_short_sentences_are_never_cut_by_the_language_model(self):
        """Real card 'Now, let's consider the specific | resistance rho' came from the LLM splitter on a 7-word sentence."""
        from app.assets import _llm_split, _split_clauses
        from app.config import AppConfig
        sentence = "Now, let's consider the specific resistance rho."
        self.assertIsNone(_llm_split(sentence, AppConfig()))          # returns before any network call
        cards = _split_clauses(sentence)
        self.assertEqual(cards[0], "Specific resistance rho")
        self.assertEqual(cards[1], "Now, let's consider the specific resistance rho")

    def test_a_natural_break_wins_over_the_language_model(self):
        """Real card: the model cut 'commercial blood | cholesterol-lowering agent' although the sentence has a colon."""
        from app import assets
        from app.assets import key_ideas
        from app.config import AppConfig
        from app.schema import Segment as Seg
        sentence = "Let's now work through question number 176, which asks: Which of the following is a commercial blood cholesterol-lowering agent?"
        calls = []
        original_split, original_post = assets._llm_split, assets.requests.post

        def offline(*args, **kwargs):
            raise assets.requests.RequestException("offline in test")

        assets._llm_split = lambda *a, **k: calls.append(1) or ["WRONG", "CUT"]
        assets.requests.post = offline                                # the card model is unreachable: the splitters decide
        try:
            cards = key_ideas(Seg(segment_number=1, narration=sentence), AppConfig())
        finally:
            assets._llm_split, assets.requests.post = original_split, original_post
        # a teacher's card is a complete idea: the key term, then the whole question, never half a sentence on each card
        self.assertEqual(cards[0], "Blood cholesterol-lowering agent")
        self.assertEqual(cards[1], sentence)
        self.assertEqual(calls, [])                                   # the model was not even asked
        self.assertIsNone(assets._split_clauses("Stay tuned to watch our upcoming videos to learn more", strong_only=True))

    def test_concept_cards_are_stored_so_a_rerun_shows_the_same_words(self):
        """The model worded the cards differently on every run: clips never stayed cached and the demo changed between runs."""
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from app import assets
        from app.schema import Segment as Seg
        owner = Seg(segment_number=1, narration="Let's begin by reading the question carefully: Two sources of equal EMF are connected.")
        job = SimpleNamespace(key="1_3")
        calls = []
        original = assets.key_ideas
        words = iter([["First wording", "Second part"], ["Different wording", "Other part"]])
        assets.key_ideas = lambda seg, cfg: calls.append(1) or next(words)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                me = SimpleNamespace(dir=Path(tmp), config=None)
                first = assets.StillProducer._card_ideas(me, job, owner)
                again = assets.StillProducer._card_ideas(me, job, owner)
                owner.narration = "A completely different sentence about something else entirely now."
                changed = assets.StillProducer._card_ideas(me, job, owner)
        finally:
            assets.key_ideas = original
        self.assertEqual(first, again)                       # reused, the model was not asked a second time
        self.assertEqual(len(calls), 2)                      # asked once, and again only after the narration changed
        self.assertEqual(changed, ["Different wording", "Other part"])

    def test_cue_text_has_no_space_before_punctuation(self):
        from app.captions import build_cues
        from app.config import AppConfig
        from app.schema import Caption
        seg = Segment(segment_number=1, narration="which is 1.", start=0.0, end=1.0, speech_duration=1.0,
                      captions=[Caption(text="which", start=0.0, end=0.3), Caption(text="is", start=0.3, end=0.5),
                                Caption(text="1", start=0.5, end=0.8), Caption(text=".", start=0.8, end=0.9)])
        cues = build_cues(seg, AppConfig().production.subtitle, (1920, 1080))
        self.assertEqual(cues[0].text, "which is 1.")

    def test_title_card_keeps_the_whole_title_but_other_headings_stay_short(self):
        title = "Minimizing the Sum of a Number and Its Reciprocal"
        card = route_segment(Segment(segment_number=1, narration="Let us work through this problem.", heading=title, visual_type="title_card"))
        self.assertEqual(card.heading, title)
        other = route_segment(Segment(segment_number=2, narration="x", heading="One two three four five six seven eight nine ten",
                                      visual_type="realistic_image"))
        self.assertEqual(len(other.heading.split()), 8)


if __name__ == "__main__":
    unittest.main()
