"""Bagian dalam pustaka ``dfjsp``; BUKAN API publik dan tidak diekspor di ``dfjsp.__all__``.

Isinya, satu modul satu urusan:

* ``datatypes``: tipe data dasar (``FJSPInstance``, ``Task``, ``Developer``, ``Schedule``).
* ``parser``: baca berkas mentah Brandimarte menjadi ``FJSPInstance``.
* ``validate``: periksa instans hasil baca.
* ``domain``: ubah instans pabrik jadi proyek perangkat lunak (kapasitas, dependensi).
* ``due_date``: tenggat (TWK) dan prioritas, serta upaya nominal ``e_nominal``.
* ``seeding``: turunan seed yang deterministik untuk tiap aliran acak.
* ``canonical``: JSON kanonis dan hash SHA-256 untuk seeding.

Kode di luar pustaka sebaiknya lewat ``dfjsp.datasets``, bukan modul ini.
"""
