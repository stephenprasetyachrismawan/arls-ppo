"""Jaringan saraf NumPy kecil (MLP dan Adam) untuk PPO pasangan."""

from __future__ import annotations

import numpy as np


def he_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    """Bobot awal He untuk lapisan ReLU."""
    return (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(np.float64)


class MLP:
    """MLP banyak lapisan: ``sizes=[masukan, h1, ..., keluaran]``.

    Lapisan tersembunyi memakai ReLU, keluaran linear. Bobot disimpan di ``params``
    dengan kunci ``W0, b0, W1, b1, ...``.

    Examples
    --------
    >>> import numpy as np
    >>> from dfjsp.selectors._nn import MLP
    >>> net = MLP([3, 4, 1], seed=0)
    >>> out, _ = net.forward(np.ones((2, 3)))
    >>> out.shape
    (2, 1)
    """

    def __init__(self, sizes: list[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.params: dict[str, np.ndarray] = {}
        for i in range(len(sizes) - 1):
            self.params[f"W{i}"] = he_init(rng, sizes[i], sizes[i + 1])
            self.params[f"b{i}"] = np.zeros(sizes[i + 1])
        self.n_layers = len(sizes) - 1

    def forward(self, X: np.ndarray):
        """Keluaran jaringan dan cache untuk ``backward``."""
        cache = {"a0": X}
        a = X
        for i in range(self.n_layers):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            cache[f"z{i}"] = z
            a = np.maximum(0.0, z) if i < self.n_layers - 1 else z
            cache[f"a{i + 1}"] = a
        return a, cache

    def backward(self, cache, dout: np.ndarray) -> dict[str, np.ndarray]:
        """Gradien semua bobot, dari turunan ``dout`` terhadap keluaran."""
        grads: dict[str, np.ndarray] = {}
        d = dout
        for i in reversed(range(self.n_layers)):
            grads[f"W{i}"] = cache[f"a{i}"].T @ d
            grads[f"b{i}"] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.params[f"W{i}"].T) * (cache[f"z{i - 1}"] > 0)
        return grads

    def param_count(self) -> int:
        """Jumlah semua bobot."""
        return int(sum(v.size for v in self.params.values()))


class Adam:
    """Optimizer Adam yang ditulis tangan."""

    def __init__(self, params, lr: float = 3e-4, betas=(0.9, 0.999), eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads) -> None:
        """Satu langkah Adam; ``params`` diubah di tempat."""
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
