"""Pemilih strategi berbasis aturan, tanpa belajar: RSR (Maintain), Random, dan Threshold."""

from __future__ import annotations

import math

import numpy as np

from ..env.state import INTENSITY, OVERLOAD
from ..repair.actions import DELAY, ESCALATE, MAINTAIN, N_ACTIONS, REASSIGN
from ..utils import BaseSelector, check_is_fitted, check_random_state
from ._common import check_instances


def _respect_mask(action: int, mask) -> int:
    """Kembalikan ``action`` bila mask mengizinkan, kalau tidak Maintain."""
    return int(action) if mask[action] == 1 else MAINTAIN


class _RuleSelector(BaseSelector):
    """Dasar pemilih aturan: ``fit`` hanya memeriksa masukan, tidak belajar apa pun."""

    def _validate_params(self) -> None:
        """Periksa parameter. Pemilih tanpa parameter tidak perlu menimpa ini."""

    def fit(self, instances):
        """Periksa parameter dan masukan, lalu tandai sudah siap.

        Parameters
        ----------
        instances : dict atau daftar instans
            Instans latih. Isinya tidak dipakai.

        Returns
        -------
        self
        """
        check_instances(instances)
        self._validate_params()
        self.is_fitted_ = True
        return self


class MaintainSelector(_RuleSelector):
    """RSR: selalu memilih Maintain (tidak mengubah jadwal).

    Examples
    --------
    >>> import numpy as np
    >>> from dfjsp.selectors import MaintainSelector
    >>> sel = MaintainSelector().fit({})
    >>> sel.predict(np.zeros(9), np.ones(4, dtype=int))
    0
    """

    def predict(self, state, mask) -> int:
        """Selalu ``Maintain`` (0)."""
        check_is_fitted(self)
        return MAINTAIN


class RandomSelector(_RuleSelector):
    """Random: memilih acak di antara strategi yang diizinkan mask.

    Parameters
    ----------
    random_state : None, int, atau numpy.random.Generator, default=None
        Sumber keacakan. ``reset(seed)`` memulai ulang urutan acak untuk satu episode.

    Examples
    --------
    >>> import numpy as np
    >>> from dfjsp.selectors import RandomSelector
    >>> sel = RandomSelector(random_state=0).fit({})
    >>> sel.reset(3)
    >>> sel.predict(np.zeros(9), np.array([1, 0, 1, 0])) in (0, 2)
    True
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    def reset(self, seed=None) -> None:
        """Mulai ulang urutan acak dari ``seed`` (bawaan: ``random_state``)."""
        self._rng = check_random_state(self.random_state if seed is None else seed)

    def predict(self, state, mask) -> int:
        """Satu strategi acak di antara yang diizinkan; Maintain bila tidak ada."""
        check_is_fitted(self)
        allowed = [a for a in range(N_ACTIONS) if mask[a] == 1]
        if not allowed:
            return MAINTAIN
        if getattr(self, "_rng", None) is None:
            self.reset()
        return int(self._rng.choice(allowed))

    def predict_proba(self, state, mask) -> np.ndarray:
        """Peluang seragam di antara strategi yang diizinkan."""
        check_is_fitted(self)
        allowed = np.array([1.0 if mask[a] == 1 else 0.0 for a in range(N_ACTIONS)])
        if allowed.sum() == 0:
            allowed[MAINTAIN] = 1.0
        return allowed / allowed.sum()


class ThresholdSelector(_RuleSelector):
    """Threshold: pemilih berbasis ambang (Persamaan 3.13).

    Aturannya, dengan ``O`` angka beban berlebih dan ``I`` angka intensitas gangguan:

    1. ``I >= tau4`` memilih Escalate;
    2. bila tidak, ``O > tau3`` memilih Reassign;
    3. bila tidak, ``tau1 <= I < tau2`` memilih Delay;
    4. selain itu Maintain.

    Pilihan yang dilarang mask diganti Maintain.

    Parameters
    ----------
    tau1, tau2 : float, default=0.2, 0.5
        Batas bawah dan atas intensitas untuk Delay.
    tau3 : float, default=0.3
        Ambang beban berlebih untuk Reassign.
    tau4 : float, default=0.8
        Ambang intensitas untuk Escalate.

    Examples
    --------
    >>> import numpy as np
    >>> from dfjsp.selectors import ThresholdSelector
    >>> state = np.zeros(9)
    >>> state[8] = 0.9          # intensitas gangguan tinggi
    >>> ThresholdSelector().fit({}).predict(state, np.ones(4, dtype=int))
    3
    """

    def __init__(self, tau1: float = 0.2, tau2: float = 0.5, tau3: float = 0.3,
                 tau4: float = 0.8):
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3
        self.tau4 = tau4

    def _validate_params(self) -> None:
        taus = {"tau1": self.tau1, "tau2": self.tau2, "tau3": self.tau3, "tau4": self.tau4}
        for name, value in taus.items():
            if not (isinstance(value, (int, float)) and math.isfinite(value)
                    and 0.0 <= value <= 1.0):
                raise ValueError(f"{name} harus bilangan di antara 0 dan 1, diterima {value!r}")
        if self.tau1 > self.tau2:
            raise ValueError(
                f"tau1 ({self.tau1}) tidak boleh lebih besar dari tau2 ({self.tau2})")

    def predict(self, state, mask) -> int:
        """Strategi menurut aturan ambang, dengan angka ``O`` dan ``I`` dari keadaan."""
        check_is_fitted(self)
        intensity = float(state[INTENSITY])
        overload = float(state[OVERLOAD])
        if intensity >= self.tau4:
            action = ESCALATE
        elif overload > self.tau3:
            action = REASSIGN
        elif self.tau1 <= intensity < self.tau2:
            action = DELAY
        else:
            action = MAINTAIN
        return _respect_mask(action, mask)
