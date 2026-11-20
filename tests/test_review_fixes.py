from __future__ import annotations

import json
import random
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.freshness.cli import run_freshness_pipeline
from tools.freshness.metrics import (
    calculate_metrics,
    compute_kaplan_meier,
    evaluate_survival_at,
)
from tools.freshness.parser import ParsedListing
from tools.freshness.state_tracker import FreshnessState, ListingRecord

DAY0 = date(2026, 1, 1)


def _day(offset: int) -> str:
    return (DAY0 + timedelta(days=offset)).isoformat()


def _record(lid: str, posted: int, last_seen: int, withdrawn: bool) -> ListingRecord:
    return ListingRecord(
        listing_id=lid,
        first_seen=_day(max(posted, 0)),
        last_seen=_day(last_seen),
        posted=_day(posted),
        status="withdrawn" if withdrawn else "active",
        withdrawn_date=_day(last_seen) if withdrawn else None,
    )


class TestKaplanMeierKnownAnswer(unittest.TestCase):
    def test_withdrawal_counts_the_day_after_it_was_last_seen(self):
        # Last seen at age 1 and 3, so withdrawn at ages 2 and 4; two listings
        # are still active at ages 5 and 2 (censored).
        records = [
            _record("A", 0, 1, True),
            _record("B", 0, 3, True),
            _record("C", 0, 5, False),
            _record("D", 3, 5, False),
        ]
        curve, median = compute_kaplan_meier(records, _day(5))
        # t=2: risk set {A, B, C, D}, 1 event -> 3/4.
        # t=4: risk set {B, C}, 1 event -> 3/4 * 1/2.
        self.assertEqual(curve, {2: 0.75, 4: 0.375})
        self.assertEqual(median, 4.0)

    def test_constant_hazard_is_unbiased_from_the_first_runs(self):
        # Daily runs, a 3% daily withdrawal hazard, listings of every age online
        # at the first run: the true S(30) is 0.97 ** 30.
        true_s30 = 0.97**30
        for runs in (2, 7):
            with self.subTest(runs=runs):
                rng = random.Random(1234)
                last_run = runs - 1
                records = []
                for i in range(40000):
                    posted = rng.randint(-200, last_run)
                    lifetime = 1
                    while rng.random() >= 0.03:
                        lifetime += 1
                    last_present = posted + lifetime - 1
                    if last_present < 0:
                        continue  # gone before the first run
                    withdrawn = last_present < last_run
                    records.append(
                        _record(str(i), posted, min(last_present, last_run), withdrawn)
                    )
                curve, _ = compute_kaplan_meier(records, _day(last_run))
                s30 = evaluate_survival_at(curve, 30)
                self.assertAlmostEqual(s30, true_s30, delta=0.03)


class TestSnapshotShares(unittest.TestCase):
    def test_age_shares_and_expiry_use_the_full_snapshot(self):
        run_date = _day(500)
        # The tracked state only holds the recent listing (window pruning).
        state = FreshnessState(
            slug="x",
            last_updated=run_date,
            listings={"new": _record("new", 490, 500, False)},
        )
        snapshot = [
            ParsedListing("new", _day(490), None, None),
            ParsedListing("old", _day(0), None, _day(600)),
            ParsedListing("mid", _day(250), None, None),
            ParsedListing("gone", _day(400), None, _day(450), operation="delete"),
        ]
        metrics = calculate_metrics(state, run_date, snapshot=snapshot)
        # Three live listings: ages 10, 500 and 250 days.
        self.assertEqual(metrics.count_older_90d, 2)
        self.assertEqual(metrics.count_older_180d, 2)
        self.assertEqual(metrics.count_older_365d, 1)
        self.assertEqual(metrics.share_older_365d, 33.33)
        self.assertEqual(metrics.active_with_expiry, 1)
        self.assertEqual(metrics.expiry_coverage_ratio, 33.33)


def _write_fixture(path: Path, ids: list[str]) -> None:
    items = [{"id": i, "datumVlozeni": "2026-09-01"} for i in ids]
    path.write_text(json.dumps({"polozky": items}), encoding="utf-8")


class TestFailedRunWritesNoState(unittest.TestCase):
    def _run(self, tmp: Path, fixture: Path, run_date: str) -> int:
        return run_freshness_pipeline(
            slug="uradprace-cz",
            input_file=fixture,
            run_date_str=run_date,
            state_dir=tmp / "freshness",
            checks_dir=tmp / "checks",
        )

    def _assert_refused(self, fixture_ids: list[str] | None, raw: str | None = None):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fixture = tmp / "data.json"
            _write_fixture(fixture, [f"L{i}" for i in range(10)])
            self.assertEqual(self._run(tmp, fixture, "2026-10-01"), 0)
            state_file = tmp / "freshness" / "uradprace-cz-state.json"
            summary_file = tmp / "freshness" / "uradprace-cz-summary.md"
            state_before = state_file.read_bytes()
            summary_before = summary_file.read_bytes()

            if raw is not None:
                fixture.write_text(raw, encoding="utf-8")
            else:
                _write_fixture(fixture, fixture_ids)
            self.assertNotEqual(self._run(tmp, fixture, "2026-10-02"), 0)

            self.assertEqual(state_file.read_bytes(), state_before)
            self.assertEqual(summary_file.read_bytes(), summary_before)
            runs = json.loads((tmp / "freshness" / "uradprace-cz-runs.json").read_text())
            self.assertEqual(runs["successful_runs"], 1)
            self.assertEqual(runs["last_failed_run"]["date"], "2026-10-02")
            return runs

    def test_zero_listings_is_refused(self):
        self._assert_refused([])

    def test_a_sharp_drop_is_refused(self):
        self._assert_refused([f"L{i}" for i in range(7)])  # 30% fewer

    def test_a_truncated_file_is_refused(self):
        self._assert_refused(None, raw='{"polozky": [{"id": "L1", "datumV')

    def test_a_small_drop_is_accepted_and_runs_are_counted(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fixture = tmp / "data.json"
            _write_fixture(fixture, [f"L{i}" for i in range(10)])
            self.assertEqual(self._run(tmp, fixture, "2026-10-01"), 0)
            _write_fixture(fixture, [f"L{i}" for i in range(9)])
            self.assertEqual(self._run(tmp, fixture, "2026-10-02"), 0)
            runs = json.loads((tmp / "freshness" / "uradprace-cz-runs.json").read_text())
            self.assertEqual(runs["successful_runs"], 2)
            self.assertIsNone(runs["last_failed_run"])
            summary = (tmp / "freshness" / "uradprace-cz-summary.md").read_text()
            self.assertIn("Successful Runs", summary)


if __name__ == "__main__":
    unittest.main()
