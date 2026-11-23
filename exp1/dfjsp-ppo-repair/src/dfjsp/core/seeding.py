"""Penurunan seed deterministik (Subbab 5.6 rancangan)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .canonical import sha256_json

_MOD = 2**32

# ---------------------------------------------------------------------------
# Protokol seed bertipe (Ladder Task 3): aliran acak dipisah secara STRUKTURAL,
# bukan lewat konvensi string "__shared__". Timeline gangguan tidak boleh
# bergantung pada metode/kandidat/arsitektur/seed-latih agar uji berpasangan
# (Wilcoxon) sah: setiap metode menghadapi lima replikasi gangguan IDENTIK per
# case identity (Subbab 3.4.2; protokol eksperimen benar).
# ---------------------------------------------------------------------------

SEED_PROTOCOL_SCHEMA = "seed-protocol-1"

SeedStream = Literal[
    "training",
    "worker",
    "domain",
    "disruption",
    "evaluation_policy",
    "bootstrap",
]

_STREAMS: frozenset[str] = frozenset(
    ("training", "worker", "domain", "disruption", "evaluation_policy", "bootstrap")
)

# Kosakata aliran bagi `derive_episode_seed` (jalur lama yang menghasilkan seluruh
# angka yang dilaporkan). Terpisah dari `_STREAMS` di atas dengan sengaja: runner
# memanggilnya dengan "policy", dan menyelaraskannya ke "evaluation_policy" akan
# mengubah setiap seed kebijakan sekaligus setiap angka baseline stokastik.
_EPISODE_STREAMS: frozenset[str] = frozenset(("domain", "disruption", "policy"))


def derive_seed(protocol_seed: int, stream: SeedStream, *identity: object) -> int:
    """Primitive derivasi seed stabil: SHA-256 atas JSON kanonis."""
    if stream not in _STREAMS:
        raise ValueError(
            f"seed stream {stream!r} tidak dikenal (sah: {sorted(_STREAMS)})"
        )
    digest = sha256_json(
        [SEED_PROTOCOL_SCHEMA, int(protocol_seed), stream, list(identity)]
    )
    return int(digest[:16], 16) % _MOD


def derive_worker_seed(training_seed: int, worker_index: int) -> int:
    """Seed worker vektor-env: unik per indeks, dapat direplikasi."""
    if worker_index < 0:
        raise ValueError(f"worker_index wajib >= 0, dapat {worker_index}")
    return derive_seed(int(training_seed), "worker", int(worker_index))


def derive_disruption_seed(
    protocol_seed: int,
    *,
    split: str,
    instance: str,
    intensity: float,
    k_tight: float,
    scenario_id: str,
    disruption_seed_index: int,
) -> int:
    """Seed timeline gangguan, INDEPENDEN dari metode/kandidat/seed-latih."""
    return derive_seed(
        int(protocol_seed),
        "disruption",
        split,
        instance,
        float(intensity),
        float(k_tight),
        scenario_id,
        int(disruption_seed_index),
    )


def derive_evaluation_policy_seed(
    protocol_seed: int,
    *,
    method_id: str,
    case_identity: tuple[object, ...],
) -> int:
    """Seed kebijakan evaluasi untuk metode stokastik (mis. random selector)."""
    return derive_seed(
        int(protocol_seed), "evaluation_policy", method_id, list(case_identity)
    )


@dataclass(frozen=True)
class CaseIdentity:
    """Identitas kasus evaluasi TANPA metode: penentu timeline gangguan."""

    split: str
    instance: str
    intensity: float
    k_tight: float
    scenario_id: str
    disruption_seed_index: int

    def as_tuple(self) -> tuple[object, ...]:
        return (
            self.split,
            self.instance,
            self.intensity,
            self.k_tight,
            self.scenario_id,
            self.disruption_seed_index,
        )


@dataclass(frozen=True)
class EvaluationCase:
    """Satu baris evaluasi: case identity + metode + seed turunan."""

    case_identity: CaseIdentity
    method_id: str
    training_seed: int | None
    disruption_seed: int
    evaluation_policy_seed: int | None


def derive_episode_seed(
    base_seed: int,
    method: str,
    instance: str,
    intensity: float,
    k_tight: float,
    seed: int,
    scenario: str,
    stream: str = "disruption",
) -> int:
    """Gabungkan kunci kombinasi menjadi satu seed integer stabil [0, 2^32).

    Parameters
    ----------
    base_seed : int
        Seed dasar penelitian.
    method, instance, scenario : str
        Nama metode, nama instans, dan nama lengkap skenario.
    intensity, k_tight : float
        Laju gangguan dan keketatan tenggat.
    seed : int
        Seed episode.
    stream : str, default="disruption"
        Pemisah aliran acak ("domain" / "disruption" / "policy") agar ketiganya tidak
        pernah tercampur meski kombinasi grid sama.

    Notes
    -----

    Nama aliran divalidasi sebab kunci ini dibangun dari ``str(x)``: salah tulis
    seperti ``stream="disrupton"`` tidak akan menimbulkan galat, hanya memberi seed
    lain yang tampak sah, dan timeline gangguan diam-diam berbeda dari yang dimaksud.
    Kosakatanya sengaja berbeda dari ``SeedStream`` protokol baru: runner memakai
    "policy", dan menyeragamkannya menjadi "evaluation_policy" akan mengubah seluruh
    seed kebijakan, yaitu seluruh angka baseline stokastik yang sudah dilaporkan.

    ``intensity`` dan ``k_tight`` dinormalkan lewat ``float`` sebelum distringkan.
    Tanpa itu ``k_tight=1`` memberi "1" sedangkan ``k_tight=1.0`` memberi "1.0",
    sehingga satu kombinasi yang secara nominal sama terpecah menjadi dua timeline
    berbeda begitu satu jalur konfigurasi membaca angka bulat tanpa desimal.
    Normalisasi ini mempertahankan seluruh seed yang berlaku sekarang, karena semua
    nilai pada configs/ sudah ditulis berdesimal (0.05, 0.15, 0.30, 1.0, 1.5, 2.0).
    """
    if stream not in _EPISODE_STREAMS:
        raise ValueError(
            f"aliran episode {stream!r} tidak dikenal (sah: {sorted(_EPISODE_STREAMS)})"
        )
    key = "|".join(
        str(x)
        for x in (stream, base_seed, method, instance,
                  float(intensity), float(k_tight), seed, scenario)
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % _MOD




#: Seed episode yang dipakai untuk menilai model: validasi 5000-5001 dan uji 10000-10009.
RESERVED_EPISODE_SEEDS: frozenset[int] = frozenset((5000, 5001, *range(10000, 10010)))


def draw_training_episode_seed(rng) -> int:
    """Seed episode untuk pelatihan, tidak pernah salah satu dari ``RESERVED_EPISODE_SEEDS``.

    Angka ditarik dari ``rng`` seperti sebelumnya (``integers(0, 2**31 - 1)``). Kalau angka
    itu seed validasi atau uji, angka berikutnya ditarik. Seed yang dulu sudah boleh tidak
    berubah, jadi pelatihan lama tetap bisa diulang persis.
    """
    while True:
        seed = int(rng.integers(0, 2**31 - 1))
        if seed not in RESERVED_EPISODE_SEEDS:
            return seed
