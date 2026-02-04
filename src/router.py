from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ConfidenceRouter:
    """
    경험적 고정 평균 대신, 샘플별 신뢰도를 기반으로 가중치를 동적으로 계산한다.
    """

    base_weights: Dict[str, float]
    confidence_mix: float = 0.6
    eps: float = 1e-8

    def _query_margin(self, scores: np.ndarray) -> np.ndarray:
        if scores.shape[1] < 2:
            return np.ones(scores.shape[0], dtype=np.float32)

        top2 = np.partition(scores, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        return np.clip(margin, 0.0, 1.0).astype(np.float32)

    def _reliability(self, scores: np.ndarray) -> np.ndarray:
        """
        pair confidence(score) + query certainty(margin) 결합.
        """
        pair_conf = np.clip(scores, 0.0, 1.0).astype(np.float32)
        query_certainty = self._query_margin(pair_conf)[:, None]
        return self.confidence_mix * pair_conf + (1.0 - self.confidence_mix) * query_certainty

    def fuse(
        self,
        scores_mega: np.ndarray,
        scores_aliked: np.ndarray,
        scores_eva: np.ndarray,
        aliked_valid_mask: np.ndarray,
    ) -> np.ndarray:
        mega_rel = self._reliability(scores_mega) * self.base_weights["mega"]
        eva_rel = self._reliability(scores_eva) * self.base_weights["eva"]

        aliked_scores_safe = np.where(aliked_valid_mask, scores_aliked, 0.0)
        aliked_rel = self._reliability(aliked_scores_safe) * self.base_weights["aliked"]
        aliked_rel = np.where(aliked_valid_mask, aliked_rel, 0.0)

        rel_sum = mega_rel + aliked_rel + eva_rel + self.eps

        w_mega = mega_rel / rel_sum
        w_aliked = aliked_rel / rel_sum
        w_eva = eva_rel / rel_sum

        return (
            w_mega * scores_mega
            + w_aliked * aliked_scores_safe
            + w_eva * scores_eva
        )
