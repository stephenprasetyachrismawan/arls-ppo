"""Pengamat gerbang 2.5: memeriksa setiap keputusan episode dengan ``dfjsp.verify``.

Dipasang di loop ``cek_episode.run`` bila ``PERIKSA=1`` (persetujuan pemilik 17 November
2026, ``docs/ARSITEKTUR_PIPELINE.md`` 5.6). Pengamat hanya membaca jadwal; ia tidak boleh
mengubah jadwal, RNG, atau keputusan metode.

Yang diperiksa di setiap keputusan:

- ``periksa_jadwal`` pada jadwal saat keputusan (sesudah gangguan) dan jadwal yang dipasang;
- ``periksa_peralihan`` dari jadwal saat keputusan ke jadwal yang dipasang;
- B4: jumlah perubahan dari metode sama dengan ``hitung_perubahan`` pemeriksa;
- B5: untuk metode berpenjaga (dan Sequential), J jadwal yang dipasang tidak lebih besar
  dari J tanpa perubahan.

Himpunan rework dikumpulkan dari gangguan ``rework_review_fail`` sejak awal episode.
"""
from __future__ import annotations

from collections import Counter

from dfjsp.verify import hitung_perubahan, periksa_jadwal, periksa_peralihan

BATAS_CONTOH = 5


class Pengamat:
    def __init__(self, berpenjaga):
        self.berpenjaga = berpenjaga
        self.rework = set()
        self.hitung = Counter()
        self.contoh = []
        self.keputusan = 0
        self.ubah_pemeriksa = 0

    def catat_gangguan(self, event):
        anggota = event.payload.get("events", []) if event.type == "batch" else [event]
        for g in anggota:
            if g.type == "rework_review_fail" and g.payload.get("task_id") is not None:
                self.rework.add(g.payload["task_id"])

    def amati(self, inst, sebelum, sesudah, t_now, perubahan, j_sebelum, j_sesudah):
        self.keputusan += 1
        temuan = [("sebelum", p) for p in periksa_jadwal(inst, sebelum, t_now, rework=self.rework)]
        temuan += [("sesudah", p) for p in periksa_jadwal(inst, sesudah, t_now, rework=self.rework)]
        temuan += [("peralihan", p) for p in periksa_peralihan(inst, sebelum, sesudah, t_now)]
        md, mp = hitung_perubahan(sebelum, sesudah, t_now)
        self.ubah_pemeriksa += md + mp
        if md + mp != perubahan:
            temuan.append(("peralihan", _Temuan("B4", f"metode {perubahan}, pemeriksa {md + mp}")))
        if self.berpenjaga and j_sesudah > j_sebelum + 1e-9:
            temuan.append(("peralihan", _Temuan("B5", f"J naik dari {j_sebelum} ke {j_sesudah}")))
        for tempat, p in temuan:
            kunci = f"{tempat}:{p.kode}:{'keras' if p.keras else 'lunak'}"
            self.hitung[kunci] += 1
            if p.keras and len(self.contoh) < BATAS_CONTOH:
                self.contoh.append({"t_now": t_now, "tempat": tempat, "kode": p.kode,
                                    "tugas": getattr(p, "tugas", None), "pesan": p.pesan})

    def ringkas(self):
        return {"keputusan_diperiksa": self.keputusan, "ubah_pemeriksa": self.ubah_pemeriksa,
                "hitung": dict(sorted(self.hitung.items())), "contoh": self.contoh}


class _Temuan:
    """Temuan B4/B5 dari pengamat; selalu keras."""

    keras = True

    def __init__(self, kode, pesan):
        self.kode, self.pesan = kode, pesan
