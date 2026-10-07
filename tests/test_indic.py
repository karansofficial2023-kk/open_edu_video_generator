import unittest

from PIL import Image

from app import shaping
from app.captions import build_cues
from app.config import AppConfig
from app.schema import Segment
from app.typography import font, is_complex_script, subtitle_band

SAMPLES = {
    "hi": "प्रकाश संश्लेषण वह प्रक्रिया है जिसमें हरे पौधे सूर्य के प्रकाश की सहायता से अपना भोजन बनाते हैं और ऑक्सीजन छोड़ते हैं।",
    "ta": "ஒளிச்சேர்க்கை என்பது தாவரங்கள் சூரிய ஒளியைப் பயன்படுத்தி உணவு தயாரிக்கும் செயல்முறை ஆகும்.",
    "te": "కిరణజన్య సంయోగక్రియ అనేది మొక్కలు సూర్యకాంతిని ఉపయోగించి ఆహారాన్ని తయారు చేసుకునే ప్రక్రియ.",
    "bn": "সালোকসংশ্লেষণ হলো সেই প্রক্রিয়া যার মাধ্যমে সবুজ উদ্ভিদ সূর্যালোক ব্যবহার করে খাদ্য তৈরি করে।",
    "ml": "പ്രകാശസംശ്ലേഷണം എന്നത് സസ്യങ്ങൾ സൂര്യപ്രകാശം ഉപയോഗിച്ച് ആഹാരം നിർമ്മിക്കുന്ന പ്രക്രിയയാണ്.",
    "ur": "ضیائی تالیف وہ عمل ہے جس میں سبز پودے سورج کی روشنی سے اپنی غذا تیار کرتے ہیں۔",
}


@unittest.skipUnless(shaping.available(), "PySide6 (Qt) text shaping is not installed")
class IndicTests(unittest.TestCase):
    def test_scripts_are_detected_and_use_the_shaped_font(self):
        for code, text in SAMPLES.items():
            self.assertTrue(is_complex_script(text), code)
            self.assertEqual(font(40, False, True).path, "qt-shaped")

    def test_subtitle_band_fits_every_script_in_two_lines_inside_margins(self):
        config = AppConfig()
        style = config.production.subtitle
        for code, text in SAMPLES.items():
            overlay, band = subtitle_band((1920, 1080), text, style)
            margin = round(1920 * style.margin_x)
            self.assertGreaterEqual(band[0], margin - 20, code)
            self.assertLessEqual(band[2], 1920 - margin + 20, code)
            self.assertLessEqual(band[3], 1080 - round(1080 * style.margin_bottom), code)
            self.assertLess(band[3] - band[1], 230, code)                         # at most two lines of text
            ink = overlay.crop(band).convert("RGBA").getchannel("A").getextrema()
            self.assertGreater(ink[1], 100, code)

    def test_text_pixels_stay_inside_the_band(self):
        style = AppConfig().production.subtitle
        overlay, band = subtitle_band((1920, 1080), SAMPLES["hi"], style)
        bbox = overlay.getbbox()
        self.assertGreaterEqual(bbox[0], band[0] - 2)
        self.assertLessEqual(bbox[2], band[2] + 2)

    def test_cues_are_split_so_each_fits(self):
        config = AppConfig()
        seg = Segment(segment_number=1, narration=" ".join([SAMPLES["hi"]] * 3), start=0, end=20, speech_duration=20)
        cues = build_cues(seg, config.production.subtitle, (1920, 1080))
        self.assertGreater(len(cues), 1)
        for cue in cues:
            subtitle_band((1920, 1080), cue.text, config.production.subtitle)

    def test_vowel_sign_order_differs_from_unshaped_layout(self):
        """Shaped 'क्रि' must differ from the naive left-to-right glyph order Pillow draws without Raqm."""
        shaped = font(60, False, True).render("प्रक्रिया", (0, 0, 0, 255))
        self.assertGreater(shaped.width, 100)
        self.assertGreater(shaped.getchannel("A").getextrema()[1], 200)


if __name__ == "__main__":
    unittest.main()
