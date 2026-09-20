"""Tests for `pythia_consensus.fusion`.

These tests are the contract for the core decision math. They must pass
without the upstream `consensus-hardening-protocol` vendored.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from pythia_consensus import ConsensusConfig, fuse
from pythia_consensus.fusion import (
    agreement_score,
    fuse_logit_mean,
    fuse_median,
    fuse_trimmed_mean,
    split_damped_weights,
    weighted_median_value,
)
from pythia_consensus.types import Estimate

def _est(
    probabilities: list[float],
    *,
    market_id: str = "mkt-1",
    analyst_prefix: str = "a",
) -> list[Estimate]:
    """Build a list of Estimate objects with given probabilities."""
    ts = datetime.now(timezone.utc).isoformat()
    return [
        Estimate(
            market_id=market_id,
            probability=p,
            confidence=0.7,
            rationale="test",
            evidence=[],
            analyst_id=f"{analyst_prefix}{i}",
            timestamp=ts,
        )
        for i, p in enumerate(probabilities)
    ]

# ---------------------------------------------------------------------------
# logit-mean
# ---------------------------------------------------------------------------
class TestLogitMean:
    def test_logit_mean_with_uniform_estimates(self) -> None:
        """When all analysts agree, logit-mean should equal that probability."""
        ests = _est([0.6, 0.6, 0.6, 0.6])
        cfg = ConsensusConfig(method="logit-mean")
        decision = fuse(ests, cfg)
        assert decision.consensus_prob == pytest.approx(0.6, abs=1e-9)
        assert decision.method == "logit-mean"

    def test_logit_mean_with_extreme_estimates(self) -> None:
        """0.01 and 0.99 should average to ~0.5 in logit space (symmetric logits)."""
        ests = _est([0.01, 0.99])
        # logit(0.01) ≈ -4.595, logit(0.99) ≈ +4.595 → mean logit ≈ 0 → sigmoid ≈ 0.5
        cfg = ConsensusConfig(method="logit-mean")
        decision = fuse(ests, cfg)
        assert decision.consensus_prob == pytest.approx(0.5, abs=1e-6)

    def test_logit_mean_direct_function(self) -> None:
        probs = np.array([0.2, 0.8])
        w = np.array([0.5, 0.5])
        # logit(0.2) ≈ -1.386, logit(0.8) ≈ +1.386 → mean logit = 0 → sigmoid = 0.5
        assert fuse_logit_mean(probs, w) == pytest.approx(0.5, abs=1e-9)

    def test_logit_mean_preserves_extreme_confidence(self) -> None:
        """3 analysts at 0.97, 1 at 0.5 — logit mean should be > arithmetic mean."""
        ests = _est([0.97, 0.97, 0.97, 0.5])
        cfg = ConsensusConfig(method="logit-mean")
        decision = fuse(ests, cfg)
        # Arithmetic mean would be (0.97*3 + 0.5)/4 = 0.8525
        # Logit mean should be higher (closer to 0.97) because the confident
        # majority pulls harder in logit space.
        assert decision.consensus_prob > 0.8525
        assert decision.consensus_prob < 0.97

# ---------------------------------------------------------------------------
# median
# ---------------------------------------------------------------------------
class TestMedian:
    def test_median(self) -> None:
        ests = _est([0.1, 0.4, 0.5, 0.6, 0.9])
        cfg = ConsensusConfig(method="median")
        decision = fuse(ests, cfg)
        # Plain median of [0.1, 0.4, 0.5, 0.6, 0.9] = 0.5
        assert decision.consensus_prob == pytest.approx(0.5, abs=1e-9)

    def test_median_robust_to_outlier(self) -> None:
        ests = _est([0.5, 0.5, 0.5, 0.5, 1.0])
        cfg = ConsensusConfig(method="median")
        decision = fuse(ests, cfg)
        # The outlier at 1.0 should not move the median off 0.5.
        assert decision.consensus_prob == pytest.approx(0.5, abs=1e-9)

    def test_median_direct_function_even_n(self) -> None:
        # Weighted median for even n with equal weights returns the lower of
        # the two middle elements (a standard weighted-median convention).
        probs = np.array([0.2, 0.4, 0.6, 0.8])
        w = np.array([0.25, 0.25, 0.25, 0.25])
        # Cumulative weights: 0.25, 0.5, 0.75, 1.0
        # searchsorted(side="left") for 0.5 returns idx=1 (cum[1]=0.5 >= 0.5).
        # → result = p_sorted[1] = 0.4
        result = fuse_median(probs, w)
        assert result == pytest.approx(0.4, abs=1e-9)

# ---------------------------------------------------------------------------
# trimmed-mean
# ---------------------------------------------------------------------------
class TestTrimmedMean:
    def test_trimmed_mean_drops_outliers(self) -> None:
        ests = _est([0.0, 0.5, 0.5, 0.5, 1.0])
        cfg = ConsensusConfig(method="trimmed-mean")
        decision = fuse(ests, cfg)
        # Drop the 0.0 and 1.0 → mean of [0.5, 0.5, 0.5] = 0.5
        assert decision.consensus_prob == pytest.approx(0.5, abs=1e-9)

    def test_trimmed_mean_falls_back_for_small_n(self) -> None:
        """For n < 3, trimmed-mean should fall back to plain weighted mean."""
        ests = _est([0.3, 0.7])
        cfg = ConsensusConfig(method="trimmed-mean")
        decision = fuse(ests, cfg)
        # Plain weighted mean of [0.3, 0.7] with equal weights = 0.5
        assert decision.consensus_prob == pytest.approx(0.5, abs=1e-9)

    def test_trimmed_mean_direct_function(self) -> None:
        probs = np.array([0.1, 0.4, 0.6, 0.9])
        w = np.array([0.25, 0.25, 0.25, 0.25])
        # Drop lowest (0.1) and highest (0.9) → [0.4, 0.6], renormalised weights [0.5, 0.5]
        # Mean = 0.5
        assert fuse_trimmed_mean(probs, w) == pytest.approx(0.5, abs=1e-9)

# ---------------------------------------------------------------------------
# agreement_score
# ---------------------------------------------------------------------------
class TestAgreementScore:
    def test_agreement_score_perfect_agreement(self) -> None:
        """All identical probabilities → agreement_score = 1.0."""
        ests = _est([0.5, 0.5, 0.5, 0.5])
        assert agreement_score(ests) == pytest.approx(1.0, abs=1e-9)

    def test_agreement_score_perfect_agreement_at_extreme(self) -> None:
        """Identical extreme probs should also give 1.0."""
        ests = _est([0.99, 0.99, 0.99])
        assert agreement_score(ests) == pytest.approx(1.0, abs=1e-9)

    def test_agreement_score_maximal_disagreement(self) -> None:
        """Half at 0, half at 1 → σ_w = 0.5 → score = 0.0."""
        ests = _est([0.0, 0.0, 1.0, 1.0])
        score = agreement_score(ests)
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_agreement_score_partial_disagreement(self) -> None:
        """[0.25, 0.75] with equal weights → σ_w = 0.25 → score = 0.5."""
        ests = _est([0.25, 0.75])
        score = agreement_score(ests)
        # σ_w = sqrt(0.5 * (0.25-0.5)^2 + 0.5 * (0.75-0.5)^2) = sqrt(0.5*0.0625*2) = sqrt(0.0625) = 0.25
        # score = 1 - 0.25/0.5 = 0.5
        assert score == pytest.approx(0.5, abs=1e-6)

# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------
class TestGateLogic:
    def test_gate_wait_when_too_few_analysts(self) -> None:
        ests = _est([0.6])  # only one analyst
        cfg = ConsensusConfig(method="logit-mean", min_analysts=2)
        decision = fuse(ests, cfg)
        assert decision.gate == "wait"

    def test_gate_wait_custom_min_analysts(self) -> None:
        ests = _est([0.6, 0.6])  # two analysts, but min=3
        cfg = ConsensusConfig(method="logit-mean", min_analysts=3)
        decision = fuse(ests, cfg)
        assert decision.gate == "wait"

    def test_gate_skip_when_low_agreement(self) -> None:
        """Maximal disagreement with enough analysts → gate = skip."""
        ests = _est([0.0, 0.0, 1.0, 1.0])  # agreement_score = 0.0
        cfg = ConsensusConfig(method="logit-mean", agreement_threshold=0.65, min_analysts=2)
        decision = fuse(ests, cfg)
        assert decision.gate == "skip"
        assert decision.agreement_score < cfg.agreement_threshold

    def test_gate_trade_when_consensus(self) -> None:
        """High agreement with enough analysts → gate = trade."""
        ests = _est([0.6, 0.62, 0.61])  # very high agreement
        cfg = ConsensusConfig(method="logit-mean", agreement_threshold=0.65, min_analysts=2)
        decision = fuse(ests, cfg)
        assert decision.gate == "trade"
        assert decision.agreement_score >= cfg.agreement_threshold

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
class TestWeights:
    def test_weights_applied(self) -> None:
        """A heavy-weighted analyst should pull the consensus toward their estimate."""
        ests = _est([0.2, 0.8])
        # Equal weights → 0.5 (logit-mean of symmetric logits).
        cfg_equal = ConsensusConfig(method="logit-mean", weights=None)
        d_equal = fuse(ests, cfg_equal)
        assert d_equal.consensus_prob == pytest.approx(0.5, abs=1e-9)

        # Heavy weight on analyst a1 (prob=0.8) → consensus should shift toward 0.8.
        cfg_heavy = ConsensusConfig(method="logit-mean", weights={"a0": 0.01, "a1": 0.99})
        d_heavy = fuse(ests, cfg_heavy)
        assert d_heavy.consensus_prob > 0.7
        assert d_heavy.weights_used == pytest.approx({"a0": 0.01, "a1": 0.99}, abs=1e-6)

    def test_weights_unknown_analyst_defaults_to_equal(self) -> None:
        """Analysts not present in the weights dict get equal (1.0) raw weight."""
        ests = _est([0.2, 0.8])
        cfg = ConsensusConfig(method="logit-mean", weights={"a0": 1.0})  # a1 missing
        d = fuse(ests, cfg)
        # Both effectively raw-weight 1.0 → normalised to 0.5 each.
        assert d.weights_used["a0"] == pytest.approx(0.5, abs=1e-9)
        assert d.weights_used["a1"] == pytest.approx(0.5, abs=1e-9)

    def test_weights_recorded_in_decision(self) -> None:
        ests = _est([0.3, 0.7, 0.5])
        cfg = ConsensusConfig(method="median", weights={"a0": 1.0, "a1": 2.0, "a2": 1.0})
        d = fuse(ests, cfg)
        assert set(d.weights_used.keys()) == {"a0", "a1", "a2"}
        # Weights should be normalised to sum to 1.
        assert sum(d.weights_used.values()) == pytest.approx(1.0, abs=1e-9)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
class TestMisc:
    def test_fuse_empty_raises(self) -> None:
        cfg = ConsensusConfig(method="logit-mean")
        with pytest.raises(ValueError, match="at least one Estimate"):
            fuse([], cfg)

    def test_decision_invariants(self) -> None:
        ests = _est([0.3, 0.7])
        cfg = ConsensusConfig(method="logit-mean")
        d = fuse(ests, cfg)
        assert 0.0 <= d.consensus_prob <= 1.0
        assert 0.0 <= d.agreement_score <= 1.0
        assert d.market_id == "mkt-1"
        assert d.contributor_ids == ["a0", "a1"]
        assert d.method == "logit-mean"
        assert d.gate in {"trade", "skip", "wait"}
        assert d.timestamp  # non-empty

class TestSplitDamping:
    """Row-1 split damping — canonical chp-core-rs v0.1.0 gate port."""

    def test_weighted_median_convention(self) -> None:
        probs = np.asarray([0.9, 0.1, 0.2])
        weights = np.asarray([1.0, 1.0, 1.0])
        assert weighted_median_value(probs, weights) == 0.2

        # Two votes: the smallest value whose cumulative weight reaches half.
        probs2 = np.asarray([0.5, 0.6])
        assert weighted_median_value(probs2, np.asarray([1.0, 1.0])) == 0.5

    def test_no_split_when_gap_small(self) -> None:
        probs = np.asarray([0.50, 0.52, 0.54, 0.55])
        weights = np.asarray([1.0, 1.0, 1.0, 1.0])
        damped, fired = split_damped_weights(probs, weights, 20.0)
        assert fired is False
        assert damped == pytest.approx(weights)

    def test_no_split_with_fewer_than_two_per_side(self) -> None:
        # A lone outlier vote must not trigger damping (canonical gate
        # requires at least two votes on each side of the gap).
        probs = np.asarray([0.10, 0.11, 0.12, 0.90])
        weights = np.asarray([1.0, 1.0, 1.0, 1.0])
        damped, fired = split_damped_weights(probs, weights, 20.0)
        assert fired is False
        assert damped == pytest.approx(weights)

    def test_split_detected_and_damped(self) -> None:
        # 2-vs-2 split with a dominant gap: damping fires (side weights are
        # unchanged when both sides are equal size, matching the canonical
        # test's expectation that the flag fires and fusion continues).
        probs = np.asarray([0.10, 0.11, 0.90, 0.91])
        weights = np.asarray([1.0, 1.0, 1.0, 1.0])
        damped, fired = split_damped_weights(probs, weights, 20.0)
        assert fired is True
        assert damped == pytest.approx(weights)

    def test_split_with_unequal_sides_damps_majority_up(self) -> None:
        # 3-vs-2 split: the larger side keeps more weight (3/5*2 = 1.2 vs
        # 2/5*2 = 0.8), so refusion pulls the fused value toward it.
        probs = np.asarray([0.10, 0.11, 0.90, 0.91, 0.92])
        weights = np.asarray([1.0] * 5)
        damped, fired = split_damped_weights(probs, weights, 20.0)
        assert fired is True
        assert fuse_logit_mean(probs, damped) > fuse_logit_mean(probs, weights)

    def test_fuse_sets_split_damped_flag(self) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        ests = [
            Estimate(market_id="mkt-1", probability=0.10, confidence=0.7,
                     rationale="r", evidence=[], analyst_id="a0", timestamp=ts),
            Estimate(market_id="mkt-1", probability=0.11, confidence=0.7,
                     rationale="r", evidence=[], analyst_id="a1", timestamp=ts),
            Estimate(market_id="mkt-1", probability=0.90, confidence=0.7,
                     rationale="r", evidence=[], analyst_id="a2", timestamp=ts),
            Estimate(market_id="mkt-1", probability=0.91, confidence=0.7,
                     rationale="r", evidence=[], analyst_id="a3", timestamp=ts),
        ]
        d = fuse(ests, ConsensusConfig(method="logit-mean"))
        assert d.split_damped is True

        # Tight distribution: no damping.
        ests_tight = [
            Estimate(market_id="mkt-1", probability=0.50 + i * 0.01,
                     confidence=0.7, rationale="r", evidence=[],
                     analyst_id=f"a{i}", timestamp=ts)
            for i in range(4)
        ]
        d2 = fuse(ests_tight, ConsensusConfig(method="logit-mean"))
        assert d2.split_damped is False


class TestDegradedGovernor:
    """Row-5 deterministic governor: degraded estimates never vote."""

    def _ests(self, degraded_flags: list[bool]) -> list[Estimate]:
        ts = datetime.now(timezone.utc).isoformat()
        probs = [0.9 if not deg else 0.5 for deg in degraded_flags]
        return [
            Estimate(market_id="mkt-1", probability=p, confidence=0.7,
                     rationale="r", evidence=[], analyst_id=f"a{i}",
                     timestamp=ts, degraded=deg)
            for i, (p, deg) in enumerate(zip(probs, degraded_flags, strict=True))
        ]

    def test_all_degraded_refuses_to_trade(self) -> None:
        d = fuse(self._ests([True, True]),
                 ConsensusConfig(method="logit-mean", min_analysts=2))
        assert d.gate == "skip"
        assert d.consensus_prob == 0.5
        assert d.governor_excluded == ["a0", "a1"]
        assert d.split_damped is False

    def test_degraded_excluded_from_fusion_and_weights(self) -> None:
        ests = self._ests([False, True])
        cfg = ConsensusConfig(method="logit-mean", min_analysts=2)
        d = fuse(ests, cfg)
        # Only one healthy analyst remains — below min_analysts, so no trade.
        assert d.gate == "wait"
        assert d.governor_excluded == ["a1"]
        assert set(d.weights_used) == {"a0"}
        # contributor_ids stays the full round roster: a1 did submit a ballot,
        # the governor just excluded it (recorded in governor_excluded).
        assert d.contributor_ids == ["a0", "a1"]
