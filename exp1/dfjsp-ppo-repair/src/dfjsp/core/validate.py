"""Validasi struktural FJSPInstance terhadap Tabel 3.2 (Tahap C, Subbab 4.3)."""

from __future__ import annotations

from .datatypes import FJSPInstance

# Tabel 3.2 proposal: (jobs, machines, operations, avg_eligible).
EXPECTED: dict[str, tuple[int, int, int, float]] = {
    "MK01": (10, 6, 55, 2.09),
    "MK02": (10, 6, 58, 4.10),
    "MK03": (15, 8, 150, 3.01),
    "MK04": (15, 8, 90, 1.91),
    "MK05": (15, 4, 106, 1.71),
    # MK06 mengikuti sumber resmi SchedulingLab/fjsp-instances (JSON & txt):
    # 10 job x 10 MESIN, 150 operasi, avg 3.27. Tabel 3.2 Bab III diselaraskan ke nilai ini.
    "MK06": (10, 10, 150, 3.27),
    "MK07": (20, 5, 100, 2.83),
    "MK08": (20, 10, 225, 1.43),
    "MK09": (20, 10, 240, 2.52),
    "MK10": (20, 15, 240, 2.98),
}

# Toleransi rata-rata eligible (Tabel 3.2 dibulatkan ke dua desimal).
AVG_TOL = 0.01


def avg_eligible(inst: FJSPInstance) -> float:
    """Rata-rata jumlah mesin eligible per operasi."""
    if not inst.tasks:
        return 0.0
    total = sum(len(t.eligible_devs) for t in inst.tasks)
    return total / len(inst.tasks)


def validate_instance(inst: FJSPInstance) -> list[str]:
    """Bandingkan struktur hasil parsing dengan EXPECTED[inst.name].

    Returns
    -------
    list of str
        Daftar pesan ketidakcocokan; daftar kosong berarti LOLOS. Pemanggil wajib
        menghentikan run bila daftar tidak kosong.
    """
    problems: list[str] = []
    if inst.name not in EXPECTED:
        problems.append(f"{inst.name}: tidak ada di Tabel 3.2 (EXPECTED).")
        return problems

    exp_jobs, exp_machines, exp_ops, exp_avg = EXPECTED[inst.name]
    n_ops = len(inst.tasks)
    got_avg = avg_eligible(inst)

    if inst.n_jobs != exp_jobs:
        problems.append(f"{inst.name}: n_jobs {inst.n_jobs} != {exp_jobs}.")
    if inst.n_devs != exp_machines:
        problems.append(f"{inst.name}: n_machines {inst.n_devs} != {exp_machines}.")
    if n_ops != exp_ops:
        problems.append(f"{inst.name}: total operasi {n_ops} != {exp_ops}.")
    if abs(round(got_avg, 2) - exp_avg) > AVG_TOL:
        problems.append(
            f"{inst.name}: avg_eligible {got_avg:.4f} (~{round(got_avg, 2)}) "
            f"!= {exp_avg} (tol {AVG_TOL})."
        )
    return problems


def matches_table_3_2(inst: FJSPInstance) -> bool:
    """Gerbang biner: True bila instans cocok Tabel 3.2 (dipakai data_manifest)."""
    return len(validate_instance(inst)) == 0
