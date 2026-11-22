"""Operator penjadwalan ulang parsial: aturan urutan dan algoritma genetika.

Keduanya menyusun ulang SEMUA tugas yang belum mulai dalam satu keputusan, berbeda dari
empat operator kanonis di ``dfjsp.repair`` yang menyentuh sedikit tugas. Perilakunya
sama dengan ``HeuristicRepair`` dan ``GAPartialRepair`` di kode lama, dibuktikan dengan
rekaman emas ``heuristics``.
"""

from __future__ import annotations

import time

import numpy as np

from ..core.datatypes import Schedule, ScheduleEntry
from ..core.due_date import e_nominal
from ..env.schedule import partition, recompute_timeline
from ..repair.base import RepairOperator, RepairResult
from ..repair.escalate import base_load
from ..repair.masks import count_feasible_pairs, is_pair_feasible
from ..utils import check_random_state

RULES = ("fifo", "spt", "lpt", "edd", "mof", "skillfit")


def order_key(rule: str):
    """Kunci urutan tugas untuk ``rule``: fungsi ``(tugas, beban_sumber) -> kunci``.

    ``beban_sumber`` dipakai ``mof`` saja: beban pengembang tempat tugas berada.
    """
    if rule == "fifo":
        return lambda task, source_load: (task.task_id,)
    if rule == "spt":
        return lambda task, source_load: (e_nominal(task), task.task_id)
    if rule == "lpt":
        return lambda task, source_load: (-e_nominal(task), task.task_id)
    if rule in ("edd", "skillfit"):
        return lambda task, source_load: (task.due_date, task.task_id)
    if rule == "mof":
        return lambda task, source_load: (-source_load, task.due_date, task.task_id)
    raise ValueError(f"rule {rule!r} tidak dikenal. Pilihan: {list(RULES)}")


def _move(sched: Schedule, task_id: int, new_dev: int) -> dict:
    """Pindahkan satu tugas ke ``new_dev`` dan kembalikan catatan pindahnya."""
    entry = sched.entries[task_id]
    sched.entries[task_id] = ScheduleEntry(task_id, new_dev, entry.start, entry.end)
    return {"task_id": task_id, "from_dev": entry.dev_id, "to_dev": new_dev}


class RuleRepair(RepairOperator):
    """Susun ulang tugas belum mulai menurut aturan urutan, lalu pilih pengembangnya.

    Tugas diurutkan dengan ``rule``, lalu satu per satu dipindah ke pengembang yang
    layak dan paling ringan bebannya. ``skillfit`` memilih pengembang yang keahliannya
    paling sedikit (paling spesialis).

    Parameters
    ----------
    rule : {"fifo", "spt", "lpt", "edd", "mof", "skillfit"}
        Aturan urutan.
    """

    def __init__(self, rule: str):
        self.rule = rule
        self.name = f"H_{rule}"

    def _pick_dev(self, candidates, loads, dev_by_id) -> int:
        if self.rule == "skillfit":
            return min(candidates,
                       key=lambda j: (len(dev_by_id[j].skills), loads.get(j, 0.0), j))
        return min(candidates, key=lambda j: (loads.get(j, 0.0), j))

    def apply(self, inst, sched: Schedule, t_now: float) -> RepairResult:
        feasible = count_feasible_pairs(inst, sched, t_now)
        _, in_progress, not_started = partition(inst, sched, t_now)
        if not not_started:
            return RepairResult(sched, [], invalid_attempt=True, feasible_pairs=feasible)

        by_id = {t.task_id: t for t in inst.tasks}
        dev_by_id = {d.dev_id: d for d in inst.developers}
        loads = base_load(inst, sched, t_now, in_progress)
        source_load = {i: loads.get(sched.entries[i].dev_id, 0.0) for i in not_started}
        key = order_key(self.rule)
        order = sorted(not_started, key=lambda i: key(by_id[i], source_load[i]))

        moved: list[dict] = []
        any_feasible = False
        for i in order:
            # ``loads`` hanya memuat tugas yang sedang berjalan. Beban tugas i belum
            # ada di dalamnya, jadi tidak ada yang dikeluarkan (exclude_self_load=False).
            candidates = [
                d.dev_id for d in inst.developers
                if is_pair_feasible(i, d.dev_id, inst, sched, t_now, loads=loads,
                                    exclude_self_load=False,
                                    by_id=by_id, dev_by_id=dev_by_id)
            ]
            if not candidates:
                continue
            any_feasible = True
            chosen = self._pick_dev(candidates, loads, dev_by_id)
            if chosen != sched.entries[i].dev_id:
                moved.append(_move(sched, i, chosen))
            loads[chosen] += by_id[i].eligible_devs[chosen]

        recompute_timeline(inst, sched, t_now)
        return RepairResult(sched, moved, invalid_attempt=not any_feasible,
                            feasible_pairs=feasible)


class GARepair(RepairOperator):
    """Algoritma genetika ringan untuk menugaskan ulang tugas belum mulai.

    Satu gen = satu tugas, nilainya pengembang. Nilai kecocokan memberi hukuman pada
    keterlambatan, ketimpangan beban, dan kelebihan kapasitas. Setelah ``apply``,
    ``last_cost`` berisi ``n_evals`` dan ``time_ms`` untuk keputusan itu.

    Parameters
    ----------
    population, generations : int
        Ukuran populasi dan jumlah generasi.
    mutation : float
        Peluang mutasi satu gen per anak.
    random_state : int atau numpy.random.Generator
        Keacakan. Seed bilangan bulat dipakai ulang di tiap ``apply``, jadi hasil satu
        keputusan hanya bergantung pada keadaan jadwal, bukan pada keputusan sebelumnya.
    """

    name = "H_GA"

    def __init__(self, population: int = 12, generations: int = 8,
                 mutation: float = 0.2, random_state=0):
        self.population = population
        self.generations = generations
        self.mutation = mutation
        self.random_state = random_state
        self.last_cost: dict = {"n_evals": 0, "time_ms": 0.0}

    def _fitness(self, genes, base, by_id, dev_by_id, sched, t_now):
        """Fungsi nilai kecocokan: ``genome -> angka`` (makin besar makin baik)."""
        def fitness(genome):
            load = dict(base)
            tardiness = 0.0
            penalty = 0.0
            for i, j in zip(genes, genome):
                entry = sched.entries[i]
                effort = by_id[i].eligible_devs.get(j, entry.end - entry.start)
                load[j] = load.get(j, 0.0) + effort
                if load[j] > dev_by_id[j].capacity + 1e-9:
                    penalty += load[j] - dev_by_id[j].capacity
                tardiness += max(0.0, (t_now + load[j]) - by_id[i].due_date)
            imbalance = float(np.std(np.array(list(load.values()), dtype=float)))
            return -(tardiness + imbalance + 100.0 * penalty)
        return fitness

    def _evolve(self, genes, choices, fitness, rng):
        """Jalankan generasi; kembalikan ``(genome_terbaik, jumlah_evaluasi)``."""
        population = [[int(rng.choice(choices[i])) for i in genes]
                      for _ in range(self.population)]
        n_evals = 0
        best, best_fit = None, -np.inf
        for _ in range(self.generations):
            scored = []
            for genome in population:
                value = fitness(genome)
                n_evals += 1
                scored.append((value, genome))
                if value > best_fit:
                    best_fit, best = value, list(genome)
            scored.sort(key=lambda pair: pair[0], reverse=True)
            elites = [g for _, g in scored[: max(2, self.population // 3)]]
            population = list(elites)
            while len(population) < self.population:
                parent_a = elites[int(rng.integers(len(elites)))]
                parent_b = elites[int(rng.integers(len(elites)))]
                cut = int(rng.integers(len(genes) + 1))
                child = parent_a[:cut] + parent_b[cut:]
                if rng.random() < self.mutation:
                    k = int(rng.integers(len(genes)))
                    child[k] = int(rng.choice(choices[genes[k]]))
                population.append(child)
        return best, n_evals

    def apply(self, inst, sched: Schedule, t_now: float) -> RepairResult:
        feasible = count_feasible_pairs(inst, sched, t_now)
        _, in_progress, not_started = partition(inst, sched, t_now)
        genes = sorted(not_started)
        if not genes:
            self.last_cost = {"n_evals": 0, "time_ms": 0.0}
            return RepairResult(sched, [], invalid_attempt=True, feasible_pairs=feasible)

        by_id = {t.task_id: t for t in inst.tasks}
        dev_by_id = {d.dev_id: d for d in inst.developers}
        base = base_load(inst, sched, t_now, in_progress)
        # Pengembang calon per gen; bila tidak ada, pengembang lama dipertahankan.
        choices = {
            i: [d.dev_id for d in inst.developers
                if d.available and d.dev_id in by_id[i].eligible_devs]
            or [sched.entries[i].dev_id]
            for i in genes
        }
        fitness = self._fitness(genes, base, by_id, dev_by_id, sched, t_now)

        started = time.perf_counter()
        best, n_evals = self._evolve(genes, choices, fitness,
                                     check_random_state(self.random_state))
        self.last_cost = {"n_evals": n_evals,
                          "time_ms": (time.perf_counter() - started) * 1000.0}

        moved = [_move(sched, i, j) for i, j in zip(genes, best)
                 if j != sched.entries[i].dev_id]
        recompute_timeline(inst, sched, t_now)
        return RepairResult(sched, moved, invalid_attempt=False, feasible_pairs=feasible)
