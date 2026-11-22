"""Pemeriksa jadwal mandiri: memeriksa apakah sebuah jadwal sah tanpa memakai kode penyusunnya.

Modul ini sengaja hanya memakai tipe data dari ``dfjsp.core.datatypes``. Pemeriksa lama
``dfjsp.env.schedule_check`` memakai fungsi penyusun jadwal yang sama dengan lingkungan,
jadi kalau fungsi itu salah, pemeriksa lama ikut salah. Di sini aturan kelayakan ditulis
ulang sendiri.
"""

from ._periksa import Pelanggaran, hitung_perubahan, periksa_jadwal, periksa_peralihan

__all__ = ["Pelanggaran", "hitung_perubahan", "periksa_jadwal", "periksa_peralihan"]
