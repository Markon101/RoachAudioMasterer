"""
ROACH EARS: Human Preference & Active Learning Engine (Layer E)
Maintains local, privacy-conscious preference dataset (data/roach_ears_preferences.jsonl),
trains Bradley-Terry pairwise skill ratings with uncertainty (mu, sigma),
and identifies highest-information-gain comparisons for active human query.
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class PreferenceRecord:
    def __init__(
        self,
        candidate_a: str,
        candidate_b: str,
        verdict: str, # "A_preferred", "B_preferred", "indistinguishable", "different_no_preference"
        confidence: float,
        track_id: str,
        excerpt_id: Optional[str] = None,
        listener_notes: Optional[str] = None,
        equipment: Optional[str] = None,
        timestamp: Optional[str] = None
    ):
        self.candidate_a = candidate_a
        self.candidate_b = candidate_b
        self.verdict = verdict
        self.confidence = confidence
        self.track_id = track_id
        self.excerpt_id = excerpt_id
        self.listener_notes = listener_notes
        self.equipment = equipment
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_a": self.candidate_a,
            "candidate_b": self.candidate_b,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "track_id": self.track_id,
            "excerpt_id": self.excerpt_id,
            "listener_notes": self.listener_notes,
            "equipment": self.equipment,
            "timestamp": self.timestamp,
        }


class PreferenceEngine:
    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path("/data/data/com.termux/files/home/projects/highband/data/roach_ears_preferences.jsonl")
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[PreferenceRecord] = []
        self.load()

    def load(self) -> None:
        self.records.clear()
        if not self.storage_path.exists():
            return
        with open(self.storage_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    self.records.append(PreferenceRecord(**d))
                except Exception:
                    pass

    def add_preference(
        self,
        candidate_a: str,
        candidate_b: str,
        verdict: str,
        confidence: float = 1.0,
        track_id: str = "general",
        excerpt_id: Optional[str] = None,
        listener_notes: Optional[str] = None,
        equipment: Optional[str] = None
    ) -> None:
        import datetime
        rec = PreferenceRecord(
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            verdict=verdict,
            confidence=confidence,
            track_id=track_id,
            excerpt_id=excerpt_id,
            listener_notes=listener_notes,
            equipment=equipment,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z"
        )
        self.records.append(rec)
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict()) + "\n")

    def fit_bradley_terry(self, num_iters: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Fits Bradley-Terry latent skill ratings (mu, sigma) with Davidson tie handling.
        Returns mapping of candidate_id -> {"skill_mu": float, "skill_sigma": float}.
        """
        # Collect unique candidates
        items = sorted(list(set(
            [r.candidate_a for r in self.records] + [r.candidate_b for r in self.records]
        )))
        if not items:
            return {}

        n = len(items)
        idx_map = {name: i for i, name in enumerate(items)}

        # Wins and comparisons matrix
        # W[i, j] = weighted count of times i beat j
        w = np.zeros((n, n), dtype=np.float64)
        ties = np.zeros((n, n), dtype=np.float64)

        for r in self.records:
            i = idx_map[r.candidate_a]
            j = idx_map[r.candidate_b]
            weight = float(r.confidence)
            if r.verdict == "A_preferred":
                w[i, j] += weight
            elif r.verdict == "B_preferred":
                w[j, i] += weight
            elif r.verdict in ("indistinguishable", "different_no_preference"):
                ties[i, j] += weight
                ties[j, i] += weight

        # Standard MM algorithm for Bradley-Terry with prior regularizer (N(0, 1))
        # gamma = exp(skill)
        gamma = np.ones(n, dtype=np.float64)

        for _ in range(num_iters):
            gamma_next = np.zeros(n, dtype=np.float64)
            for i in range(n):
                wins_i = np.sum(w[i, :]) + 0.5 * np.sum(ties[i, :]) + 1.0 # Laplace regularizer
                denom = 0.0
                for j in range(n):
                    if i == j:
                        continue
                    n_ij = w[i, j] + w[j, i] + ties[i, j]
                    if n_ij > 0:
                        denom += n_ij / (gamma[i] + gamma[j])
                denom += 1.0 # prior anchor
                gamma_next[i] = wins_i / denom
            # Normalize product or sum
            gamma = gamma_next / np.mean(gamma_next)

        skills = np.log(np.maximum(1e-12, gamma))
        # Anchor mean skill to 0.0
        skills -= np.mean(skills)

        # Approximate posterior standard deviation using observed Fisher information
        ratings = {}
        for i, name in enumerate(items):
            info = 1.0 # prior precision
            for j in range(n):
                if i != j and (w[i, j] + w[j, i] + ties[i, j]) > 0:
                    n_ij = w[i, j] + w[j, i] + ties[i, j]
                    p = gamma[i] / (gamma[i] + gamma[j])
                    info += n_ij * p * (1.0 - p)
            sigma = float(1.0 / np.sqrt(info))
            ratings[name] = {
                "skill_mu": round(float(skills[i]), 3),
                "skill_sigma": round(sigma, 3),
                "comparisons_count": int(np.sum(w[i, :] + w[:, i] + ties[i, :]))
            }

        return ratings

    def recommend_active_comparison(
        self,
        candidate_pool: List[str]
    ) -> Optional[Tuple[str, str, float]]:
        """
        Active Learning: Selects the pair (cand_a, cand_b) that maximizes expected
        information gain (pairs where skill ratings are closest and uncertainty is highest).
        """
        if len(candidate_pool) < 2:
            return None

        ratings = self.fit_bradley_terry()
        best_pair = None
        max_info_gain = -1.0

        for i in range(len(candidate_pool)):
            for j in range(i + 1, len(candidate_pool)):
                a, b = candidate_pool[i], candidate_pool[j]
                mu_a = ratings.get(a, {}).get("skill_mu", 0.0)
                sig_a = ratings.get(a, {}).get("skill_sigma", 1.0)
                mu_b = ratings.get(b, {}).get("skill_mu", 0.0)
                sig_b = ratings.get(b, {}).get("skill_sigma", 1.0)

                # Pair uncertainty = sqrt(sig_a^2 + sig_b^2)
                pair_uncertainty = np.sqrt(sig_a ** 2 + sig_b ** 2)
                # Proximity of skill: closer skill = higher ambiguity = higher informative value
                skill_diff = abs(mu_a - mu_b)
                # Information gain heuristic: entropy of prediction * joint uncertainty
                prob_a_beats_b = 1.0 / (1.0 + np.exp(-(mu_a - mu_b)))
                entropy = -(prob_a_beats_b * np.log2(max(1e-6, prob_a_beats_b)) +
                            (1.0 - prob_a_beats_b) * np.log2(max(1e-6, 1.0 - prob_a_beats_b)))
                info_gain = entropy * pair_uncertainty

                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_pair = (a, b, float(info_gain))

        return best_pair
