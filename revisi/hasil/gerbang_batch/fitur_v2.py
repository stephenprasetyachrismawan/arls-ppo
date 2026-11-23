"""Fitur v2 (E2): semua angka tanpa satuan dan tidak bergantung ukuran instans.

Fitur lama (``cek_b3.features``, 38 angka) masih memuat jumlah mentah (banyak tugas
terdampak, banyak penerima sah, banyak pasangan) dan beban dalam satuan waktu. Instans besar
seperti MK10 lalu terlihat seperti keadaan yang tidak pernah dilihat. Di sini:

- waktu dibagi E (rata-rata effort) atau sisa jadwal (makespan - t);
- jumlah tugas dibagi banyak tugas terdampak atau banyak tugas yang belum selesai;
- jumlah pengembang dibagi banyak pengembang.

Ditambah 6 fitur murah yang tidak butuh simulasi (kajian 4.2, D7): sisa jadwal, waktu kosong
pengembang (minimum, median, maksimum), waktu kosong penerima sah terbaik, dan selisih beban
pemilik tugas terdampak dengan penerima sah paling ringan. Waktu kosong dan selisih beban
ditulis sebagai bagian dari sisa jadwal, bukan dalam satuan E, karena sisa jadwal dalam
satuan E ikut membesar pada proyek yang panjang.

Panjang vektor tetap 43, untuk instans berapa pun ukurannya.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from dfjsp.env.schedule import developer_load, partition
from dfjsp.env.state import makespan, total_tardiness
from dfjsp.repair.masks import is_pair_feasible

TYPES = ("urgent_task", "effort_revision", "rework_review_fail", "priority_change",
         "developer_unavailable")
DIM = 43


def features(env, cur, base_state, j, A, types, lam, changes_so_far, ops):
    """Fitur v2 untuk gangguan ke-j pada jadwal ``cur``. Kembalikan (vektor, mask 3 bit)."""
    inst, t, E = env.inst, env.t_now, env.mean_effort
    _, in_prog, not_started = partition(inst, cur, t)
    ns = set(not_started)
    by_id = {x.task_id: x for x in inst.tasks}
    dev_by_id = {d.dev_id: d for d in inst.developers}
    n_dev = max(1, len(inst.developers))
    loads = developer_load(inst, cur, t)
    unfinished = max(1, len(in_prog) + len(not_started))
    rest_span = max(E, makespan(inst, cur) - t)       # sisa jadwal, paling kecil satu E

    a_j = A[j]
    n_a = max(1, len(a_j))
    a_ns = [x for x in a_j if x in ns]
    late = [x for x in a_j if cur.entries[x].end > by_id[x].due_date + 1e-9]
    tard_a = sum(max(0.0, cur.entries[x].end - by_id[x].due_date) for x in a_j) / E
    slack = [(by_id[x].due_date - cur.entries[x].end) / E for x in a_ns] or [0.0]
    n_elig = [len(by_id[x].eligible_devs) / n_dev for x in a_ns] or [0.0]

    recv, n_pairs = set(), 0
    for x in a_ns:
        curdev = cur.entries[x].dev_id
        for d in by_id[x].eligible_devs:
            if d != curdev and is_pair_feasible(x, d, inst, cur, t, loads=loads,
                                                by_id=by_id, dev_by_id=dev_by_id):
                recv.add(d)
                n_pairs += 1
    trial = copy.copy(cur)
    trial.affected = set(a_j)
    mask = [int(ops[lv].inspect(inst, trial, t).feasible) for lv in (1, 2, 3)]

    # Utilisasi seperti fitur lama: kapasitas = kelipatan beban rata-rata awal, jadi
    # beban / kapasitas sudah tanpa satuan.
    util = np.array([loads.get(d.dev_id, 0.0) / max(1e-9, d.capacity)
                     for d in inst.developers], dtype=float)
    # Waktu kosong tiap pengembang sampai ujung jadwal, sebagai bagian dari sisa jadwal
    # (0 = penuh sampai ujung, 1 = kosong sama sekali).
    idle = {d.dev_id: max(0.0, 1.0 - loads.get(d.dev_id, 0.0) / rest_span)
            for d in inst.developers}
    iv = np.array(list(idle.values()), dtype=float)
    owners = {cur.entries[x].dev_id for x in a_ns}
    best_recv_idle = max((idle[d] for d in recv), default=0.0)
    load_owner = float(np.mean([loads.get(d, 0.0) for d in owners])) if owners else 0.0
    load_recv = min((loads.get(d, 0.0) for d in recv), default=load_owner)

    others = set().union(*[A[i] for i in range(len(A)) if i != j]) if len(A) > 1 else set()
    rest = set().union(*A[j + 1:]) if j + 1 < len(A) else set()
    k = len(A)
    f = list(base_state) + [
        *[1.0 if types[j] == ty else 0.0 for ty in TYPES],
        len(a_j) / unfinished, len(a_ns) / n_a,
        len(late) / n_a, math.log1p(tard_a / n_a),
        float(np.mean(slack)), float(np.min(slack)),
        float(np.mean(n_elig)), len(recv) / n_dev, n_pairs / (max(1, len(a_ns)) * n_dev),
        *mask,
        float(np.mean(util)), float(np.max(util)), float(np.std(util)),
        math.log1p(total_tardiness(inst, cur) / E / unfinished),
        k, j / k, (k - j - 1),
        len(set(a_j) & others) / n_a, len(rest) / unfinished,
        changes_so_far / n_a, math.log(lam),
        # Fitur murah tambahan (kajian 4.2).
        rest_span / max(E, makespan(inst, cur)),
        float(iv.min()), float(np.median(iv)), float(iv.max()),
        best_recv_idle,
        (load_owner - load_recv) / rest_span,
    ]
    assert len(f) == DIM, len(f)
    return np.asarray(f, dtype=np.float32), mask
