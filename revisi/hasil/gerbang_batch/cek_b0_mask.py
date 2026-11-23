"""Gerbang B0 + pemeriksaan dua lapis mask untuk rencana per-gangguan dalam batch.

Hanya instans latih (BRANDIMARTE_TRAIN), seed validasi 5000-5001. Jalur episode selalu
tingkat 0 (AOR), sama seperti uji oracle. Tidak ada pelatihan, data uji tidak disentuh.

Per kejadian batch dicatat:
- k, jenis anggota;
- A_j: tugas yang langsung dikenai d_j atau terdorong terlambat saat d_j diterapkan
  (anggota diterapkan berurutan, persis seperti apply_disruption);
- mask tingkat atas per gangguan (tingkat mana yang punya calon bila A = A_j);
- mask tingkat bawah: pengembang penerima yang sah untuk tugas A_j, dan apakah dua
  gangguan berebut penerima yang sama padahal sisa kapasitasnya tidak cukup untuk keduanya;
- pemotongan: tingkat 2 batch hanya menyentuh 3 tugas pertama dari gabungan A.
"""
from __future__ import annotations

import penjaga_uji  # noqa: F401  (memblokir pembukaan data uji)

import copy
import json
import sys
from collections import Counter
from itertools import combinations
from multiprocessing import Pool

import numpy as np

SRC = sys.argv[1]
OUT = sys.argv[2]
sys.path.insert(0, SRC)

from dfjsp.core.domain import set_capacity  # noqa: E402
from dfjsp.datasets import BRANDIMARTE_TRAIN, load_brandimarte  # noqa: E402
from dfjsp.env.disruption_effects import apply_disruption  # noqa: E402
from dfjsp.env.ladder_env import LadderRepairEnv  # noqa: E402
from dfjsp.env.repair_env import _direct_targets, affected_tasks  # noqa: E402
from dfjsp.env.schedule import developer_load, partition  # noqa: E402
from dfjsp.env.state import total_tardiness  # noqa: E402
from dfjsp.repair.ladder import ladder_operators, _light_copy  # noqa: E402
from dfjsp.repair.masks import is_pair_feasible  # noqa: E402

CAPS = (1.0, 1.5, 2.0)
INTENSITIES = (0.05, 0.15, 0.30)
SEEDS = (5000, 5001)


class Probe(LadderRepairEnv):
    """Lingkungan tangga yang juga mencatat A_j per anggota batch."""

    def _apply_event(self, event):
        self.per_sub = None
        if event.type != "batch":
            return super()._apply_event(event)
        tard_before = total_tardiness(self.inst, self.sched)
        before_all = {tid: e.end for tid, e in self.sched.entries.items()}
        by_id = None
        per = []
        for sub in event.payload.get("events", []):
            before = {tid: e.end for tid, e in self.sched.entries.items()}
            apply_disruption(self.inst, self.sched, sub, self.t_now, self.refs.horizon)
            by_id = {t.task_id: t for t in self.inst.tasks}
            direct = _direct_targets(sub, before, self.sched, self.t_now)
            late = {tid for tid, e in self.sched.entries.items()
                    if tid in before and e.end > before[tid] + 1e-9
                    and e.end > by_id[tid].due_date + 1e-9}
            per.append((sub.type, {t for t in direct | late if t in self.sched.entries}))
        self.sched.affected = affected_tasks(self.inst, self.sched, event, before_all,
                                             self.t_now)
        self._event_added = total_tardiness(self.inst, self.sched) - tard_before
        self.per_sub = per
        self._probe()


def analyse(env: Probe) -> dict:
    inst, sched, t = env.inst, env.sched, env.t_now
    _, _, not_started = partition(inst, sched, t)
    ns = set(not_started)
    by_id = {x.task_id: x for x in inst.tasks}
    dev_by_id = {d.dev_id: d for d in inst.developers}
    loads = developer_load(inst, sched, t)
    ops = ladder_operators(enforce_capacity=True)
    union_ns = {x for x in sched.affected if x in ns}

    subs = []
    for typ, a_j in env.per_sub:
        a_ns = sorted(x for x in a_j if x in ns)
        trial = _light_copy(sched)
        trial.affected = set(a_j)
        allowed = [1] + [int(ops[lv].inspect(inst, trial, t).feasible) for lv in (1, 2, 3)]
        # Penerima sah (lapis bawah, kapasitas diperiksa) per tugas A_j yang belum mulai.
        recv = {}
        for i in a_ns:
            cur = sched.entries[i].dev_id
            for j in by_id[i].eligible_devs:
                if j != cur and is_pair_feasible(i, j, inst, sched, t, loads=loads,
                                                 by_id=by_id, dev_by_id=dev_by_id):
                    recv.setdefault(j, []).append(by_id[i].eligible_devs[j])
        subs.append({"type": typ, "n_A": len(a_j), "n_A_ns": len(a_ns),
                     "allowed": allowed, "recv": recv, "tasks": set(a_ns)})

    # Interaksi lewat lapis bawah: dua gangguan berebut penerima yang sisa kapasitasnya
    # tidak cukup untuk satu tugas dari masing-masing.
    shared, tight, overlap = 0, 0, 0
    for s1, s2 in combinations(subs, 2):
        if s1["tasks"] & s2["tasks"]:
            overlap += 1
        common = set(s1["recv"]) & set(s2["recv"])
        if common:
            shared += 1
            for r in common:
                spare = dev_by_id[r].capacity - loads.get(r, 0.0)
                if spare + 1e-9 < min(s1["recv"][r]) + min(s2["recv"][r]):
                    tight += 1
                    break
    k = len(subs)
    n_allowed = [sum(s["allowed"]) for s in subs]
    return {
        "k": k,
        "types": [s["type"] for s in subs],
        "n_allowed": n_allowed,
        "combos_eff": int(np.prod(n_allowed)),
        "combos_full": 4 ** k,
        "n_real_choice": sum(1 for n in n_allowed if n > 1),
        "n_A_ns": [s["n_A_ns"] for s in subs],
        "union_ns": len(union_ns),
        "batch_mask": [int(env.probes.get(a) is not None) for a in (0, 1, 2, 3)],
        "pairs": k * (k - 1) // 2,
        "pairs_overlap": overlap,
        "pairs_shared_recv": shared,
        "pairs_tight": tight,
    }


def run(job):
    name, cap, inten, seed = job
    inst = copy.deepcopy(load_brandimarte([name])[name])
    set_capacity(inst, cap)
    env = Probe(lam=1.0)
    env.reset(inst, intensity=inten, seed=seed)
    rows, n_dec, n_single = [], 0, 0
    while not env.done:
        n_dec += 1
        if getattr(env, "per_sub", None):
            r = analyse(env)
            r.update(instans=name, kapasitas=cap, intensitas=inten, seed=seed)
            rows.append(r)
        else:
            n_single += 1
        env.step(0)
    return rows, n_dec, n_single


if __name__ == "__main__":
    jobs = [(n, c, i, s) for n in BRANDIMARTE_TRAIN for c in CAPS for i in INTENSITIES
            for s in SEEDS]
    with Pool(8) as pool:
        res = pool.map(run, jobs)
    rows = [r for rs, _, _ in res for r in rs]
    meta = {"keputusan": sum(x[1] for x in res), "tunggal": sum(x[2] for x in res),
            "batch": len(rows), "episode": len(jobs)}
    with open(OUT, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f)
    print(json.dumps(meta))
