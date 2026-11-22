"""Pemeriksa kelayakan jadwal yang berdiri sendiri."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.datatypes import FJSPInstance, Schedule
from .schedule import developer_load, partition
from .work_calendar import working_time

EPS = 1e-6


@dataclass(frozen=True)
class HasilPemeriksaanJadwal:
    """Daftar pelanggaran yang ditemukan pada satu jadwal pada waktu tertentu."""

    keahlian: tuple[tuple[int, int], ...]
    ketergantungan: tuple[tuple[int, int], ...]
    tumpang_tindih: tuple[tuple[int, int, int], ...]
    tugas_tanpa_jadwal: tuple[int, ...]
    kapasitas: tuple[tuple[int, float, float], ...]
    upaya_kurang: tuple[int, ...] = ()

    @property
    def layak_keras(self) -> bool:
        """Kendala yang tidak boleh dilanggar oleh jadwal mana pun."""
        return not (
            self.keahlian
            or self.ketergantungan
            or self.tumpang_tindih
            or self.tugas_tanpa_jadwal
            or self.upaya_kurang
        )

    @property
    def ada_kelebihan_beban(self) -> bool:
        """Kondisi yang boleh muncul, tetapi harus dilaporkan, bukan disembunyikan."""
        return bool(self.kapasitas)

    def ringkas(self) -> str:
        """Satu baris untuk pesan galat tes."""
        return (
            f"keahlian={list(self.keahlian)} "
            f"ketergantungan={list(self.ketergantungan)} "
            f"tumpang_tindih={list(self.tumpang_tindih)} "
            f"tugas_tanpa_jadwal={list(self.tugas_tanpa_jadwal)} "
            f"kapasitas={list(self.kapasitas)} "
            f"upaya_kurang={list(self.upaya_kurang)}"
        )


def check_schedule(inst: FJSPInstance, sched: Schedule,
                   t_now: float) -> HasilPemeriksaanJadwal:
    """Periksa satu jadwal pada waktu `t_now` tanpa memakai mask.

    Parameters
    ----------
    inst : FJSPInstance
        Instans berisi tugas, developer, keahlian, dan kapasitas.
    sched : Schedule
        Jadwal yang diperiksa.
    t_now : float
        Waktu pemeriksaan; menentukan tugas mana yang masih dapat digeser.

    Returns
    -------
    HasilPemeriksaanJadwal
        Daftar pelanggaran per jenis. Kosong pada semua jenis berarti jadwal memenuhi
        kendala keras dan tidak ada developer yang kelebihan beban.
    """
    by_id = {t.task_id: t for t in inst.tasks}
    _, _, belum_mulai = partition(inst, sched, t_now)

    tugas_tanpa_jadwal = tuple(
        sorted(tid for tid in by_id if sched.entries.get(tid) is None)
    )

    keahlian = tuple(
        (tid, entry.dev_id)
        for tid, entry in sorted(sched.entries.items())
        if entry.dev_id not in by_id[tid].eligible_devs
    )

    selesai = {tid: entry.end for tid, entry in sched.entries.items()}
    # Tugas yang tidak punya entri sudah dilaporkan tersendiri; ia tidak punya
    # waktu mulai sehingga ketergantungannya tidak dapat dinilai di sini.
    ketergantungan = tuple(
        (tid, dep)
        for tid in sorted(belum_mulai)
        if sched.entries.get(tid) is not None
        for dep in by_id[tid].deps
        if dep in selesai and selesai[dep] > sched.entries[tid].start + EPS
    )

    per_dev: dict[int, list[tuple[float, float, int]]] = {}
    for tid, entry in sched.entries.items():
        per_dev.setdefault(entry.dev_id, []).append((entry.start, entry.end, tid))
    tumpang_tindih = []
    for dev_id in sorted(per_dev):
        rentang = sorted(per_dev[dev_id])
        for (_, akhir_a, tid_a), (mulai_b, _, tid_b) in zip(rentang, rentang[1:]):
            if akhir_a > mulai_b + EPS:
                tumpang_tindih.append((dev_id, tid_a, tid_b))

    beban = developer_load(inst, sched, t_now)
    kapasitas = tuple(
        (dev.dev_id, float(beban.get(dev.dev_id, 0.0)), float(dev.capacity))
        for dev in sorted(inst.developers, key=lambda d: d.dev_id)
        if beban.get(dev.dev_id, 0.0) > dev.capacity + EPS
    )

    return HasilPemeriksaanJadwal(
        keahlian=keahlian,
        ketergantungan=ketergantungan,
        tumpang_tindih=tuple(tumpang_tindih),
        tugas_tanpa_jadwal=tugas_tanpa_jadwal,
        kapasitas=kapasitas,
        # Revisi turun boleh membuat pekerjaan lampau lebih besar daripada
        # estimasi baru. Waktu efektif yang KURANG tetap tidak layak.
        upaya_kurang=tuple(tid for tid, e in sorted(sched.entries.items())
            if e.dev_id in by_id[tid].eligible_devs
            and working_time(e.start, e.end, sched.pauses.get(e.dev_id, [])) + EPS
                < by_id[tid].eligible_devs[e.dev_id]),
    )


def ringkas_kelebihan_beban(inst: FJSPInstance, sebelum: dict[int, float],
                            sesudah: dict[int, float]) -> dict[str, float]:
    """Bandingkan beban sebelum dan sesudah satu tindakan pada waktu yang sama."""
    hasil = {
        "overload_devs_before": 0.0,
        "overload_devs_after": 0.0,
        "overload_excess_before": 0.0,
        "overload_excess_after": 0.0,
        "overload_excess_added": 0.0,
    }
    for dev in inst.developers:
        cap = float(dev.capacity)
        lebih_sebelum = max(0.0, float(sebelum.get(dev.dev_id, 0.0)) - cap)
        lebih_sesudah = max(0.0, float(sesudah.get(dev.dev_id, 0.0)) - cap)
        hasil["overload_devs_before"] += 1.0 if lebih_sebelum > EPS else 0.0
        hasil["overload_devs_after"] += 1.0 if lebih_sesudah > EPS else 0.0
        hasil["overload_excess_before"] += lebih_sebelum
        hasil["overload_excess_after"] += lebih_sesudah
        hasil["overload_excess_added"] += max(0.0, lebih_sesudah - lebih_sebelum)
    return hasil
