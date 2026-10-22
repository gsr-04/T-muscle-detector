"""Minimal public-layout and formula tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    def test_public_entrypoints(self):
        expected = (
            "tmuscle.py",
            "pipeline/00_test_image.py",
            "pipeline/07_detect_usable_arms.py",
            "pipeline/09_analyze_abdomen.py",
            "README.md",
            "LICENSE",
        )
        for relative in expected:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_default_scoring_config(self):
        config = json.loads(
            (ROOT / "configs" / "default.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["distance_weight"], 1.3)
        self.assertAlmostEqual(
            sum(config["total_weights"].values()), 1.0
        )
        self.assertEqual(
            config["abdomen_score"]["pivot_angle_deg"], 240.0
        )
        self.assertEqual(config["abdomen_score"]["span_deg"], 180.0)
        self.assertEqual(
            config["shoulder_waist_score"]["pivot"], 1.3
        )
        self.assertEqual(
            config["shoulder_waist_score"]["span"], 0.8
        )

    def test_legacy_flows_removed(self):
        self.assertFalse((ROOT / "final").exists())
        tools = list((ROOT / "tools").glob("*.py"))
        self.assertLessEqual(len(tools), 30)


if __name__ == "__main__":
    unittest.main()
