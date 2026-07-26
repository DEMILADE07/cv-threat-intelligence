from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from customization import CustomizationEngine, RawEvent


class CompoundRecipeTests(unittest.TestCase):
    def test_compound_recipe_fires_from_two_medium_signals(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            cfg = Path(raw_tmp) / "compound.json"
            cfg.write_text(
                """
                {
                  "use_case_id": "compound",
                  "rules": [
                    {
                      "name": "armed_robbery",
                      "title": "ARMED ROBBERY IN PROGRESS",
                      "priority": "critical",
                      "signals": ["weapon_candidate", "violence"],
                      "logic": "one_high_or_two_medium",
                      "gate_question": "Do these frames show an armed robbery?"
                    }
                  ]
                }
                """
            )
            engine = CustomizationEngine(cfg)

            alerts = engine.evaluate(
                [
                    RawEvent(detector="weapons", active=True, title="WEAPON", level="medium"),
                    RawEvent(detector="violence", active=True, title="VIOLENCE", level="medium"),
                ]
            )

            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0].rule_name, "armed_robbery")
            self.assertEqual(alerts[0].detector, "compound")
            self.assertEqual(alerts[0].title, "ARMED ROBBERY IN PROGRESS")
            self.assertEqual(alerts[0].question, "Do these frames show an armed robbery?")

    def test_compound_recipe_does_not_fire_on_one_medium_signal(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            cfg = Path(raw_tmp) / "compound.json"
            cfg.write_text(
                '{"rules":[{"name":"armed_robbery","signals":["weapon_candidate","violence"],"logic":"one_high_or_two_medium"}]}'
            )
            engine = CustomizationEngine(cfg)

            alerts = engine.evaluate(
                [RawEvent(detector="weapons", active=True, title="WEAPON", level="medium")]
            )

            self.assertEqual(alerts, [])


if __name__ == "__main__":
    unittest.main()
