from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from customization import CustomizationEngine, RawEvent


class BaselineCriticalTests(unittest.TestCase):
    def test_baseline_rules_are_evaluated_with_customer_config(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            baseline = tmp / "baseline.json"
            customer = tmp / "customer.json"
            baseline.write_text(
                '{"use_case_id":"baseline","rules":[{"name":"baseline_weapon","trigger":{"detector":"weapons"},"priority":"critical"}]}'
            )
            customer.write_text(
                '{"use_case_id":"customer","rules":[{"name":"shoplifting","trigger":{"detector":"concealment"},"priority":"high"}]}'
            )
            engine = CustomizationEngine(customer, baseline_path=baseline)

            alerts = engine.evaluate(
                [
                    RawEvent(detector="weapons", active=True, title="GUN", level="critical"),
                    RawEvent(detector="concealment", active=True, title="CONCEALMENT", level="high"),
                ]
            )

            self.assertEqual([alert.rule_name for alert in alerts], ["baseline_weapon", "shoplifting"])

    def test_additional_rule_file_appends_recipes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            customer = tmp / "customer.json"
            recipes = tmp / "recipes.json"
            customer.write_text('{"rules":[{"name":"shoplifting","trigger":{"detector":"concealment"},"priority":"high"}]}')
            recipes.write_text('{"rules":[{"name":"violent_theft","signals":["concealment","violence"],"logic":"at_least_2","priority":"high"}]}')
            engine = CustomizationEngine(customer)

            engine.load_additional(recipes)
            alerts = engine.evaluate(
                [
                    RawEvent(detector="concealment", active=True, title="CONCEAL", level="high"),
                    RawEvent(detector="violence", active=True, title="VIOLENCE", level="medium"),
                ]
            )

            self.assertIn("shoplifting", [alert.rule_name for alert in alerts])
            self.assertIn("violent_theft", [alert.rule_name for alert in alerts])


if __name__ == "__main__":
    unittest.main()
