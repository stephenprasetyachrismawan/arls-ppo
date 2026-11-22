"""Pembanding pencarian per gangguan: algoritma genetika atas penugasan dan urutan.

Menggantikan GA-PR lama sebagai pembanding "mencari ulang setiap gangguan" (D7 di
``revisi/diagnosis_metode_sekarang.md``). Perbedaan dengan GA-PR:

* memutuskan **penugasan dan urutan** (GA-PR hanya penugasan);
* menilai calon dengan jadwal yang benar-benar dihitung ulang (GA-PR memakai perkiraan
  beban kumulatif);
* memakai tujuan yang sama dengan agen tangga: keterlambatan / E + lambda x perubahan.

Kromosom untuk ``k`` tugas yang dikeluarkan (kumpulan yang sama dengan tingkat 3): satu
pengembang per tugas dan satu urutan penyisipan. Dekode: tugas disisipkan menurut urutan
itu, masing-masing di posisi menurut tenggat pada antrean pengembang pilihannya.
"""

from __future__ import annotations

import numpy as np

from ..env.schedule import developer_load, queue_order, recompute_timeline
from ..env.state import total_tardiness
from .base import RepairOperator, RepairResult
from .ladder import PartialReschedule, _changes, _edd_insert, _light_copy, _moves, set_queue
from .masks import is_pair_feasible


class GALadder(RepairOperator):
    """Algoritma genetika ringan di atas kumpulan tugas tingkat 3.

    Parameters
    ----------
    lam : float
        Harga satu perubahan (sama dengan agen).
    mean_effort : float
        Satuan keterlambatan E.
    population, generations : int, default=12, 10
    mutation : float, default=0.3
    max_tasks : int, default=12
    seed : int, default=0
        Seed tetap: hasil satu keputusan hanya bergantung pada keadaan jadwal.
    """

    name = "GA"

    def __init__(self, lam: float, mean_effort: float, population: int = 12,
                 generations: int = 10, mutation: float = 0.3, max_tasks: int = 12,
                 seed: int = 0, enforce_capacity: bool = True):
        self.lam = lam
        self.mean_effort = mean_effort
        self.population = population
        self.generations = generations
        self.mutation = mutation
        self.max_tasks = max_tasks
        self.seed = seed
        self.enforce_capacity = enforce_capacity
        self.last_evals = 0

    def apply(self, inst, sched, t_now):
        pool = PartialReschedule(max_tasks=self.max_tasks)._pool(inst, sched, t_now)
        if not pool:
            return RepairResult(sched, [], invalid_attempt=True)
        rng = np.random.default_rng(self.seed)
        by_id = {t.task_id: t for t in inst.tasks}
        dev_by_id = {d.dev_id: d for d in inst.developers}
        loads = developer_load(inst, sched, t_now)
        choices = []
        for tid in pool:
            home = sched.entries[tid].dev_id
            ok = [j for j in sorted(by_id[tid].eligible_devs)
                  if j == home or is_pair_feasible(tid, j, inst, sched, t_now, loads=loads,
                                                   enforce_capacity=self.enforce_capacity,
                                                   by_id=by_id, dev_by_id=dev_by_id)]
            choices.append(ok or [home])
        base_queues = {j: [t for t in q if t not in set(pool)]
                       for j, q in queue_order(inst, sched, t_now).items()}
        cache: dict = {}
        self.last_evals = 0

        def decode(genome):
            devs, order = genome
            key = (tuple(devs), tuple(order))
            if key in cache:
                return cache[key]
            queues = {j: list(q) for j, q in base_queues.items()}
            for k in order:
                j = devs[k]
                queues[j] = _edd_insert(queues.get(j, []), pool[k], by_id)
            trial = _light_copy(sched)
            for j, q in queues.items():
                if q:
                    set_queue(trial, j, q, t_now)
            recompute_timeline(inst, trial, t_now)
            moved_dev, moved_pos = _changes(sched, trial, t_now, inst)
            cost = total_tardiness(inst, trial) / self.mean_effort + self.lam * (
                moved_dev + moved_pos)
            self.last_evals += 1
            cache[key] = (cost, trial)
            return cache[key]

        n = len(pool)
        home = [sched.entries[t].dev_id for t in pool]
        edd = sorted(range(n), key=lambda k: (by_id[pool[k]].due_date, pool[k]))
        popn = [(list(home), list(edd))]   # calon awal: tanpa pindah, urutan tenggat
        while len(popn) < self.population:
            popn.append(([int(rng.choice(c)) for c in choices], list(rng.permutation(n))))
        for _ in range(self.generations):
            scored = sorted(popn, key=lambda g: decode(g)[0])
            elite = scored[: max(2, self.population // 3)]
            popn = list(elite)
            while len(popn) < self.population:
                a = elite[int(rng.integers(len(elite)))]
                b = elite[int(rng.integers(len(elite)))]
                cut = int(rng.integers(n + 1))
                devs = a[0][:cut] + b[0][cut:]
                # Urutan: crossover berurutan (OX sederhana) lalu mutasi tukar.
                head = a[1][:cut]
                order = head + [k for k in b[1] if k not in set(head)]
                if rng.random() < self.mutation:
                    i = int(rng.integers(n))
                    devs[i] = int(rng.choice(choices[i]))
                if rng.random() < self.mutation and n > 1:
                    i, j = rng.choice(n, size=2, replace=False)
                    order[i], order[j] = order[j], order[i]
                popn.append((devs, order))
        best_cost, best = min((decode(g) for g in popn), key=lambda x: x[0])
        base_cost = total_tardiness(inst, sched) / self.mean_effort
        if best_cost >= base_cost - 1e-9:
            return RepairResult(sched, [], invalid_attempt=True)
        before = _light_copy(sched)
        sched.entries = best.entries
        recompute_timeline(inst, sched, t_now)
        return RepairResult(sched, _moves(before, sched), invalid_attempt=False)
