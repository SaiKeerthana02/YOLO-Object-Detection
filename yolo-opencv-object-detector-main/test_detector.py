"""Small tests for the parts of detector.py that do not need a camera."""

from collections import Counter
import unittest

from detector import class_name, format_counts, parse_source


class DetectorHelpersTest(unittest.TestCase):
    def test_numeric_source_becomes_camera_index(self) -> None:
        self.assertEqual(parse_source("0"), 0)
        self.assertEqual(parse_source(" 2 "), 2)

    def test_video_path_remains_text(self) -> None:
        self.assertEqual(parse_source("videos/demo.mp4"), "videos/demo.mp4")

    def test_class_name_supports_dictionary_and_list(self) -> None:
        self.assertEqual(class_name({0: "person"}, 0), "person")
        self.assertEqual(class_name(["person", "bicycle"], 1), "bicycle")
        self.assertEqual(class_name({}, 99), "class_99")

    def test_counts_are_sorted(self) -> None:
        counts = Counter({"person": 2, "bottle": 1})
        self.assertEqual(format_counts(counts), "bottle: 1 | person: 2")


if __name__ == "__main__":
    unittest.main()
