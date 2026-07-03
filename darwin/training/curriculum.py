"""Design-space curriculum: expand the sampling box as the policy improves."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Curriculum:
    c: float
    c_final: float
    c_step: float
    return_threshold: float
    check_every_iters: int
    _returns: list = dataclasses.field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: dict) -> "Curriculum":
        return cls(
            c=float(cfg["c_init"]),
            c_final=float(cfg["c_final"]),
            c_step=float(cfg["c_step"]),
            return_threshold=float(cfg["return_threshold"]),
            check_every_iters=int(cfg["check_every_iters"]),
        )

    def record(self, mean_episode_return: float):
        self._returns.append(float(mean_episode_return))

    def maybe_expand(self, iteration: int) -> bool:
        """Every `check_every_iters`, widen the box if returns clear the bar."""
        if iteration == 0 or iteration % self.check_every_iters != 0:
            return False
        window = self._returns[-self.check_every_iters:]
        self._returns = []
        if window and sum(window) / len(window) >= self.return_threshold:
            old = self.c
            self.c = min(self.c_final, self.c + self.c_step)
            return self.c > old
        return False
