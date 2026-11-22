"""Pemuat angka hasil ARLS-PPO untuk artikel dan dek (seam disetujui pemilik 22 November 2026).

Artikel Springer dan dek pendadaran tidak menghitung ulang apa pun. Keduanya membaca angka dari
JSON yang ditulis ``bab5_jalan.py`` ke ``latex/TemplateLatexLaporanTesis/chapters/chapter5/
generated/``, lewat kelas :class:`HasilArls`. Dengan begitu artikel, dek, dan Bab V memakai
angka dari berkas yang sama.

Modul ini hanya memakai pustaka bawaan Python, supaya dapat diimpor oleh lingkungan Python
artikel maupun dek.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: Berkas yang dibaca :meth:`HasilArls.baca`, semuanya keluaran ``bab5_jalan.py``.
BERKAS = {"ringkas": "bab5_uji_ringkas.json", "uraian": "bab5_uji_uraian.json",
          "aor": "bab5_uji_aor.json", "validasi": "bab5_validasi_tahap.json",
          "kuartil": "bab5_uji_kuartil.json", "latih": "bab5_latih_bobot.json",
          "episode": "bab5_episode.json"}


@dataclass(frozen=True)
class Selang:
    """Titik dan batas selang sebuah selisih dalam persen (negatif = ARLS-PPO lebih baik)."""

    titik: float
    bawah: float
    atas: float

    @property
    def tegas(self) -> bool:
        """Benar bila selang tidak memuat nol."""
        return self.atas < 0 or self.bawah > 0


def _kunci_lam(lam: float) -> str:
    """Kunci λ di JSON: ``str(float)``, misalnya 0.3 -> "0.3" dan 3 -> "3.0"."""
    return str(float(lam))


class HasilArls:
    """Angka bernama dari JSON hasil Bab V.

    ``data`` = kamus berisi isi berkas JSON, dengan kunci seperti pada :data:`BERKAS`.
    """

    def __init__(self, data: dict) -> None:
        self._d = data

    @classmethod
    def baca(cls, folder) -> "HasilArls":
        """Baca semua berkas pada :data:`BERKAS` dari ``folder`` (folder ``generated`` Bab V)."""
        folder = Path(folder)
        data = {}
        for kunci, nama in BERKAS.items():
            p = folder / nama
            if not p.exists():
                raise FileNotFoundError(f"{nama} tidak ada di {folder}; jalankan bab5_jalan.py")
            data[kunci] = json.loads(p.read_text(encoding="utf-8"))
        return cls(data)

    def banding(self, pembanding: str, lam: float, ukuran: str,
                keluarga: str = "gabungan") -> Selang:
        """Selisih J ARLS-PPO terhadap ``pembanding`` (persen) pada ``lam``.

        ``ukuran`` = "uk1" atau "uk2"; ``keluarga`` = "utama" (selang 98,75%), "gabungan" atau
        "benchmark_saja" (selang 95%).
        """
        x = self._d["ringkas"][keluarga][pembanding][_kunci_lam(lam)][ukuran]
        return Selang(*x)

    def kalimat(self, pembanding: str, lam: float) -> str:
        """Kalimat klaim butir 2 menurut ``docs/KUNCI_UJI.md`` bagian 7, dari selang utama.

        "lebih baik" atau "lebih buruk" hanya bila kedua ukuran searah dan tegas; selain itu
        "tidak tegas".
        """
        u1 = self.banding(pembanding, lam, "uk1", keluarga="utama")
        u2 = self.banding(pembanding, lam, "uk2", keluarga="utama")
        if u1.atas < 0 and u2.atas < 0:
            return "lebih baik"
        if u1.bawah > 0 and u2.bawah > 0:
            return "lebih buruk"
        return "tidak tegas"

    def rentang(self, pembanding: str, ukuran: str, keluarga: str = "gabungan") -> tuple:
        """(terkecil, terbesar) titik selisih terhadap ``pembanding`` atas semua λ yang ada."""
        titik = [v[ukuran][0] for v in self._d["ringkas"][keluarga][pembanding].values()]
        return min(titik), max(titik)

    def waktu(self, metode: str) -> dict:
        """Waktu keputusan p95 dalam milidetik: pemilih terkecil dan terbesar atas instans uji,
        dan waktu total terbesar. Metode tanpa pemilih terpisah memberi None untuk pemilih."""
        w = self._d["ringkas"]["waktu"][metode]
        per = list(w["p95_pemilih_per_instans"].values())
        return {"pemilih_min": min(per) if per else None,
                "pemilih_maks": max(per) if per else None,
                "total_maks": w["p95_total_maks"]}

    def pelanggaran(self) -> dict:
        """Jumlah episode dan pelanggaran keras pemeriksa jadwal atas semua metode."""
        p = self._d["ringkas"]["pelanggaran"].values()
        return {"episode": sum(v["episode"] for v in p), "keras": sum(v["keras"] for v in p)}

    @staticmethod
    def _cari(baris: list, metode: str, lam: float) -> dict:
        for r in baris:
            if r["metode"] == metode and abs(r["lam"] - lam) < 1e-9:
                return r
        raise KeyError(f"tidak ada baris {metode} pada lambda {lam}")

    def uraian(self, metode: str, lam: float) -> dict:
        """Suku J pada data uji (rata-rata per instans lalu atas instans): telat, ubah, J."""
        r = self._cari(self._d["uraian"], metode, lam)
        return {"telat": r["telat"], "ubah": r["ubah"], "J": r["J"]}

    def tahap(self, dari: str, ke: str, lam: float) -> float:
        """Perubahan J dalam persen dari tahap ``dari`` ke tahap ``ke`` pada data validasi."""
        a = self._cari(self._d["validasi"], dari, lam)["J"]
        b = self._cari(self._d["validasi"], ke, lam)["J"]
        return (b / a - 1) * 100

    def butir1(self, pasangan: str, ukuran: str) -> Selang:
        """Selisih celah ke Sequential Lookahead (poin) untuk satu pasangan λ butir 1."""
        return Selang(*self._d["ringkas"]["butir1"][pasangan][ukuran])

    def perubahan(self, pembanding: str, lam: float) -> Selang:
        """Selisih banyak perubahan per episode, ARLS-PPO dikurangi pembanding (selang 95%)."""
        return Selang(*self._d["ringkas"]["stabil"][pembanding][_kunci_lam(lam)])

    def aor(self, metode: str, lam: float, ukuran: str) -> Selang:
        """Selisih ``metode`` terhadap AOR (selang 95%).

        ``ukuran`` = "uk1" atau "uk2" (selisih J dalam persen), "telat" (selisih keterlambatan
        dalam satuan E), atau "ubah" (selisih banyak perubahan per episode).
        """
        d = self._d["aor"][metode]
        k = _kunci_lam(lam)
        if ukuran in ("uk1", "uk2"):
            return Selang(*d["J"][k][ukuran])
        return Selang(*d[ukuran][k])

    def kurva(self, metode: str) -> list:
        """Titik (λ, keterlambatan, banyak perubahan) satu metode pada data uji, urut menurut λ."""
        return sorted((r["lam"], r["telat"], r["ubah"]) for r in self._d["uraian"]
                      if r["metode"] == metode)

    def kuartil(self, lam: float, kelompok: int) -> dict:
        """Satu baris pemeriksaan kuartil ARLS-PPO lawan Fixed-L3 + penjaga (``bab5_uji_kuartil``).

        Kelompok 1 sampai 4 menurut J Fixed-L3 + penjaga di dalam setiap instans (Q1 terkecil).
        """
        for r in self._d["kuartil"][_kunci_lam(lam)]:
            if r["kelompok"] == kelompok:
                return r
        raise KeyError(f"tidak ada kelompok {kelompok} pada lambda {lam}")

    def bobot(self, bagian: str) -> tuple:
        """(terkecil, terbesar) perubahan relatif bobot dari sebelum ke sesudah tahap PPO.

        ``bagian`` = "actor" (lapis ``policy_net`` dan ``action_net``) atau "critic" (lapis
        ``value_net``), atas semua seed. Lapis yang norma awalnya nol (None) dilewati.
        """
        cocok = {"actor": ("policy_net", "action_net"), "critic": ("value_net",)}[bagian]
        nilai = [x["ubah_relatif"] for daftar in self._d["latih"]["bobot"].values()
                 for x in daftar if x["ubah_relatif"] is not None
                 and any(c in x["lapis"] for c in cocok)]
        return min(nilai), max(nilai)

    def pilih(self, kunci: str) -> tuple:
        """(terkecil, terbesar) atas seed porsi kesamaan pilihan pada keadaan data guru.

        ``kunci`` = "sama" (sebelum = sesudah PPO), "sama_guru_a" (sebelum = guru), atau
        "sama_guru_b" (sesudah = guru).
        """
        nilai = [r[kunci] for r in self._d["latih"]["pilih"].values()]
        return min(nilai), max(nilai)

    def episode(self, lam: float) -> dict:
        """Ringkasan episode contoh (MK01, satu seed model) pada ``lam``.

        ``diminta`` = sub-keputusan yang meminta tingkat 1 sampai 3; ``tidak_dipakai`` = yang di
        antaranya akhirnya memakai tingkat 0 (ditolak penjaga atau operatornya gagal).
        """
        e = self._d["episode"]["lam"][_kunci_lam(lam)]
        pasang = [(a, b) for k in e["kejadian"] for a, b in zip(k["diminta"], k["dipakai"])]
        return {"kejadian": len(e["kejadian"]), "sub_keputusan": len(pasang),
                "diminta": sum(a > 0 for a, _ in pasang),
                "tidak_dipakai": sum(a > 0 and b == 0 for a, b in pasang),
                "pulih": sum(k["pulih"] for k in e["kejadian"]), "ubah": e["ubah"],
                "telat_akhir": e["telat_akhir"], "J_akhir": e["J_akhir"]}
