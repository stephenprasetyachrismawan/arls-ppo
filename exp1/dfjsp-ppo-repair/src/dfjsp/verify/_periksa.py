"""Aturan kelayakan jadwal, ditulis tanpa memakai ``dfjsp.env`` atau ``dfjsp.repair``."""

from __future__ import annotations

import math
from dataclasses import dataclass

# Selisih waktu (jam) yang masih dianggap sama, supaya galat pembulatan float tidak
# terbaca sebagai pelanggaran.
TOLERANSI = 1e-6


@dataclass(frozen=True)
class Pelanggaran:
    """Satu pelanggaran aturan jadwal.

    Attributes
    ----------
    kode : str
        Kode aturan: ``A1`` sampai ``A8`` untuk satu jadwal, ``B1`` sampai ``B4`` untuk
        satu keputusan, ``K1`` untuk kapasitas.
    tugas : int or None
        Nomor tugas yang melanggar, None bila pelanggaran milik pengembang.
    pengembang : int or None
        Nomor pengembang yang terlibat, None bila tidak ada.
    pesan : str
        Penjelasan singkat beserta angkanya.
    keras : bool
        True berarti jadwal tidak sah. False berarti dilaporkan saja (kapasitas, atau
        urutan pendahulu yang dilanggar karena rework).

    Examples
    --------
    >>> p = Pelanggaran("A3", 1, 0, "pengembang tidak punya keahlian untuk tugas ini")
    >>> p.kode, p.keras
    ('A3', True)
    """

    kode: str
    tugas: int | None
    pengembang: int | None
    pesan: str
    keras: bool = True


def _gabung_jeda(jeda):
    """Jeda yang bertumpuk digabung supaya tidak dihitung dua kali."""
    gabung = []
    for a, b in sorted((a, b) for a, b in jeda if b > a):
        if gabung and a <= gabung[-1][1]:
            gabung[-1] = (gabung[-1][0], max(gabung[-1][1], b))
        else:
            gabung.append((a, b))
    return gabung


def _lama_kerja(mulai, selesai, jeda):
    """Jam kerja di [mulai, selesai) dikurangi bagian yang jatuh di dalam jeda."""
    berhenti = sum(max(0.0, min(selesai, b) - max(mulai, a)) for a, b in _gabung_jeda(jeda))
    return selesai - mulai - berhenti


def periksa_jadwal(inst, jadwal, t_now, *, rework=()):
    """Daftar pelanggaran pada satu jadwal; daftar kosong berarti jadwal sah.

    Tugas dengan mulai <= ``t_now`` dianggap sudah jalan. Aturannya:

    - A1 tiap tugas instans punya tepat satu entri, dan tidak ada entri tugas asing
    - A2 waktu mulai dan selesai berhingga, selesai >= mulai
    - A3 pengembang punya keahlian untuk tugas (ada di ``eligible_devs``)
    - A4 satu pengembang tidak mengerjakan dua tugas sekaligus, batas [mulai, selesai)
    - A5 tugas mulai sesudah semua pendahulunya selesai
    - A6 lama kerja tanpa jeda sama dengan upaya (tugas belum mulai) atau tidak kurang
      dari upaya (tugas sudah jalan)
    - A7 tugas belum mulai tidak mulai di dalam jeda pengembangnya
    - A8 tugas belum mulai tidak mulai sebelum waktu rilisnya
    - K1 (lunak) sisa kerja pengembang sesudah ``t_now`` tidak melebihi kapasitas

    Parameters
    ----------
    inst : FJSPInstance
        Instans dengan upaya terbaru (sesudah gangguan).
    jadwal : Schedule
        Jadwal yang diperiksa.
    t_now : float
        Waktu sekarang.
    rework : set of int, optional
        Tugas yang pernah kena rework. Untuk tugas yang sudah jalan, pelanggaran A5
        terhadap pendahulu di himpunan ini dilaporkan lunak, karena rework mengembalikan
        pendahulu menjadi belum mulai sesudah penerusnya jalan.

    Returns
    -------
    list of Pelanggaran
        Urut menurut aturan. Kosong berarti sah.

    Examples
    --------
    >>> from dfjsp.core.datatypes import (Developer, FJSPInstance, Schedule,
    ...                                   ScheduleEntry, Task)
    >>> inst = FJSPInstance("kecil", 1, 1, [Task(0, 0, 0, {0: 2.0}), Task(1, 0, 1, {0: 1.0}, deps=[0])],
    ...                     [Developer(0, capacity=10.0)])
    >>> jadwal = Schedule({0: ScheduleEntry(0, 0, 1.0, 3.0), 1: ScheduleEntry(1, 0, 3.0, 4.0)})
    >>> periksa_jadwal(inst, jadwal, 0.0)
    []
    >>> jadwal.entries[1] = ScheduleEntry(1, 0, 2.0, 3.0)   # mulai sebelum tugas 0 selesai
    >>> sorted({p.kode for p in periksa_jadwal(inst, jadwal, 0.0)})
    ['A4', 'A5']
    """
    hasil = []
    tugas = {t.task_id: t for t in inst.tasks}
    for tid in sorted(tugas):
        if tid not in jadwal.entries:
            hasil.append(Pelanggaran("A1", tid, None, "tugas tidak punya entri di jadwal"))
    for tid in sorted(jadwal.entries):
        if tid not in tugas:
            hasil.append(Pelanggaran("A1", tid, jadwal.entries[tid].dev_id,
                                     "entri untuk tugas yang tidak ada di instans"))
    sah = {}
    for tid in sorted(set(tugas) & set(jadwal.entries)):
        e = jadwal.entries[tid]
        if not (math.isfinite(e.start) and math.isfinite(e.end)) or e.end < e.start:
            hasil.append(Pelanggaran("A2", tid, e.dev_id,
                                     f"waktu tidak wajar: mulai {e.start}, selesai {e.end}"))
        else:
            sah[tid] = e
    for tid, e in sah.items():
        if e.dev_id not in tugas[tid].eligible_devs:
            hasil.append(Pelanggaran("A3", tid, e.dev_id,
                                     "pengembang tidak punya keahlian untuk tugas ini"))
    per_dev = {}
    for tid, e in sah.items():
        if e.end > e.start:
            per_dev.setdefault(e.dev_id, []).append((e.start, e.end, tid))
    for dev in sorted(per_dev):
        selesai_terjauh, pemilik = -math.inf, None
        for mulai, selesai, tid in sorted(per_dev[dev]):
            if mulai < selesai_terjauh - TOLERANSI:
                hasil.append(Pelanggaran("A4", tid, dev,
                                         f"bertumpuk dengan tugas {pemilik} (sampai {selesai_terjauh})"))
            if selesai > selesai_terjauh:
                selesai_terjauh, pemilik = selesai, tid
    for tid, e in sorted(sah.items()):
        for dep in tugas[tid].deps:
            if dep in sah and sah[dep].end > e.start + TOLERANSI:
                # Tugas yang sudah jalan boleh mendahului pendahulu yang kena rework:
                # rework mengembalikan pendahulu menjadi belum mulai sesudah penerusnya jalan.
                diterima = e.start <= t_now and dep in rework
                hasil.append(Pelanggaran("A5", tid, e.dev_id,
                                         f"mulai {e.start} sebelum pendahulu {dep} selesai "
                                         f"({sah[dep].end})", keras=not diterima))
    for tid, e in sorted(sah.items()):
        upaya = tugas[tid].eligible_devs.get(e.dev_id)
        if upaya is None:
            continue  # sudah dilaporkan A3
        kerja = _lama_kerja(e.start, e.end, jadwal.pauses.get(e.dev_id, []))
        # Tugas yang sudah jalan boleh lebih lama dari upayanya (waktu yang sudah lewat
        # tidak dihapus saat estimasi diturunkan); tugas yang belum mulai harus pas.
        if e.start <= t_now:
            salah = kerja < upaya - TOLERANSI
        else:
            salah = abs(kerja - upaya) > TOLERANSI
        if salah:
            hasil.append(Pelanggaran("A6", tid, e.dev_id,
                                     f"lama kerja {kerja} jam, upaya {upaya} jam"))
    for tid, e in sorted(sah.items()):
        if e.start <= t_now:
            continue
        for a, b in _gabung_jeda(jadwal.pauses.get(e.dev_id, [])):
            if a <= e.start < b:
                hasil.append(Pelanggaran("A7", tid, e.dev_id,
                                         f"mulai {e.start} di dalam jeda [{a}, {b})"))
        rilis = jadwal.release.get(tid)
        if rilis is not None and e.start < rilis - TOLERANSI:
            hasil.append(Pelanggaran("A8", tid, e.dev_id,
                                     f"mulai {e.start} sebelum waktu rilis {rilis}"))
    # Kapasitas adalah kendala lunak: kelebihan beban dilaporkan, tetapi tidak membuat
    # jadwal tidak sah. Beban = sisa jam kerja sesudah t_now dari tugas yang belum selesai.
    sisa = {}
    for tid, e in sah.items():
        if e.end > t_now:
            jam = _lama_kerja(max(e.start, t_now), e.end, jadwal.pauses.get(e.dev_id, []))
            sisa[e.dev_id] = sisa.get(e.dev_id, 0.0) + jam
    for d in sorted(inst.developers, key=lambda d: d.dev_id):
        if sisa.get(d.dev_id, 0.0) > d.capacity + TOLERANSI:
            hasil.append(Pelanggaran("K1", None, d.dev_id,
                                     f"sisa kerja {sisa[d.dev_id]} jam melebihi kapasitas "
                                     f"{d.capacity} jam", keras=False))
    return hasil


def periksa_peralihan(inst, sebelum, sesudah, t_now, *, perubahan=None):
    """Daftar pelanggaran pada satu keputusan perbaikan, dari ``sebelum`` ke ``sesudah``.

    Fungsi ini hanya memeriksa peralihannya. Kelayakan ``sesudah`` sendiri diperiksa
    dengan :func:`periksa_jadwal`. Aturannya:

    - B1 tugas yang sudah jalan di ``sebelum`` (mulai <= ``t_now``) tidak berubah
    - B2 himpunan tugas dan jeda pengembang tidak berubah oleh keputusan
    - B3 tugas yang belum mulai di ``sebelum`` tidak dipindah ke sebelum ``t_now``
      (dimajukan tepat ke ``t_now`` boleh)
    - B4 hitungan perubahan yang dilaporkan sama dengan :func:`hitung_perubahan`

    Parameters
    ----------
    inst : FJSPInstance
        Instans.
    sebelum, sesudah : Schedule
        Jadwal sebelum dan sesudah keputusan perbaikan.
    t_now : float
        Waktu keputusan.
    perubahan : tuple of int, optional
        Hitungan (pindah pengembang, pindah posisi) dari kode yang diperiksa. Bila tidak
        diberikan, B4 tidak diperiksa.

    Returns
    -------
    list of Pelanggaran

    Examples
    --------
    >>> from dfjsp.core.datatypes import (Developer, FJSPInstance, Schedule,
    ...                                   ScheduleEntry, Task)
    >>> inst = FJSPInstance("kecil", 1, 1, [Task(0, 0, 0, {0: 2.0}), Task(1, 0, 1, {0: 1.0}, deps=[0])],
    ...                     [Developer(0, capacity=10.0)])
    >>> jadwal = Schedule({0: ScheduleEntry(0, 0, 1.0, 3.0), 1: ScheduleEntry(1, 0, 3.0, 4.0)})
    >>> import copy
    >>> sesudah = copy.deepcopy(jadwal)
    >>> sesudah.entries[0] = ScheduleEntry(0, 0, 0.5, 2.5)   # dimajukan ke masa lalu
    >>> [p.kode for p in periksa_peralihan(inst, jadwal, sesudah, 0.8)]
    ['B3']
    """
    hasil = []
    for tid, lama in sorted(sebelum.entries.items()):
        baru = sesudah.entries.get(tid)
        if lama.start <= t_now and not _sama(lama, baru):
            hasil.append(Pelanggaran("B1", tid, lama.dev_id,
                                     f"tugas sudah jalan diubah: {_teks(lama)} menjadi "
                                     f"{_teks(baru)}"))
        if lama.start > t_now and baru is not None and baru.start < t_now - TOLERANSI:
            hasil.append(Pelanggaran("B3", tid, baru.dev_id,
                                     f"tugas belum mulai dipindah ke masa lalu: mulai "
                                     f"{baru.start} < t_now {t_now}"))
    for tid in sorted(set(sebelum.entries) ^ set(sesudah.entries)):
        ada = "hilang" if tid in sebelum.entries else "muncul"
        hasil.append(Pelanggaran("B2", tid, None, f"tugas {ada} karena keputusan perbaikan"))
    for dev in sorted(set(sebelum.pauses) | set(sesudah.pauses)):
        if _gabung_jeda(sebelum.pauses.get(dev, [])) != _gabung_jeda(sesudah.pauses.get(dev, [])):
            hasil.append(Pelanggaran("B2", None, dev, "jeda pengembang berubah"))
    if perubahan is not None:
        hitungan = hitung_perubahan(sebelum, sesudah, t_now)
        if tuple(perubahan) != hitungan:
            hasil.append(Pelanggaran("B4", None, None,
                                     f"hitungan perubahan dilaporkan {tuple(perubahan)}, "
                                     f"hitungan pemeriksa {hitungan}"))
    return hasil


def _sama(a, b):
    return (b is not None and a.dev_id == b.dev_id and abs(a.start - b.start) <= TOLERANSI
            and abs(a.end - b.end) <= TOLERANSI)


def _teks(e):
    return "tidak ada" if e is None else f"pengembang {e.dev_id} [{e.start}, {e.end}]"


def hitung_perubahan(sebelum, sesudah, t_now):
    """(tugas pindah pengembang, tugas pindah posisi) dari ``sebelum`` ke ``sesudah``.

    Pindah pengembang: tugas yang pengembangnya berbeda. Pindah posisi: pada tugas yang
    belum mulai di ``sebelum`` dan tetap di pengembang yang sama, banyak tugas paling
    sedikit yang harus dipindah supaya antrean lama menjadi antrean baru, yaitu n dikurangi
    panjang barisan naik terpanjang. Antrean diurutkan menurut waktu mulai, lalu nomor
    tugas. Tugas yang dimajukan tepat ke ``t_now`` tetap dihitung.

    Parameters
    ----------
    sebelum, sesudah : Schedule
        Jadwal sebelum dan sesudah keputusan.
    t_now : float
        Waktu keputusan.

    Returns
    -------
    tuple of int
        ``(pindah_pengembang, pindah_posisi)``.

    Examples
    --------
    >>> from dfjsp.core.datatypes import (Developer, FJSPInstance, Schedule,
    ...                                   ScheduleEntry, Task)
    >>> inst = FJSPInstance("kecil", 1, 1, [Task(0, 0, 0, {0: 2.0}), Task(1, 0, 1, {0: 1.0}, deps=[0])],
    ...                     [Developer(0, capacity=10.0)])
    >>> jadwal = Schedule({0: ScheduleEntry(0, 0, 1.0, 3.0), 1: ScheduleEntry(1, 0, 3.0, 4.0)})
    >>> import copy
    >>> sesudah = copy.deepcopy(jadwal)
    >>> sesudah.entries[0] = ScheduleEntry(0, 0, 2.0, 4.0)   # tugas 1 dimajukan ke depan
    >>> sesudah.entries[1] = ScheduleEntry(1, 0, 1.0, 2.0)
    >>> hitung_perubahan(jadwal, sesudah, 0.0)
    (0, 1)
    """
    pindah_dev = sum(1 for tid, e in sesudah.entries.items()
                     if tid in sebelum.entries and sebelum.entries[tid].dev_id != e.dev_id)
    menunggu = [tid for tid, e in sebelum.entries.items() if e.start > t_now]
    lama = _antrean(sebelum, menunggu)
    baru = _antrean(sesudah, [t for t in menunggu if t in sesudah.entries])
    pindah_posisi = 0
    for dev, antrean in baru.items():
        posisi_lama = {t: k for k, t in enumerate(lama.get(dev, []))}
        urutan = [posisi_lama[t] for t in antrean if t in posisi_lama]
        pindah_posisi += len(urutan) - _naik_terpanjang(urutan)
    return (pindah_dev, pindah_posisi)


def _antrean(jadwal, daftar):
    """Antrean tiap pengembang untuk tugas di ``daftar``, urut waktu mulai lalu nomor tugas."""
    antrean = {}
    for tid in sorted(daftar, key=lambda t: (jadwal.entries[t].start, t)):
        antrean.setdefault(jadwal.entries[tid].dev_id, []).append(tid)
    return antrean


def _naik_terpanjang(urutan):
    """Panjang barisan naik terpanjang, cara sederhana O(n^2) supaya mudah diperiksa."""
    panjang = []
    for i, x in enumerate(urutan):
        panjang.append(1 + max((panjang[j] for j in range(i) if urutan[j] < x), default=0))
    return max(panjang, default=0)
