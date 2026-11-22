"""Alat bantu kecil yang dipakai semua pemilih."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..datasets import BRANDIMARTE_TEST


def check_instances(instances) -> dict:
    """Ubah masukan ``fit`` menjadi kamus nama instans -> instans.

    Parameters
    ----------
    instances : dict atau daftar instans
        Kamus nama -> instans (misalnya hasil ``split_instances``), atau daftar
        instans yang punya atribut ``name``.

    Raises
    ------
    ValueError
        Bila ``instances`` bukan kamus atau daftar.

    Examples
    --------
    >>> from dfjsp.selectors._common import check_instances
    >>> check_instances({"MK01": 1})
    {'MK01': 1}
    """
    if instances is None or isinstance(instances, (str, bytes)):
        raise ValueError(
            "instances harus berupa kamus nama -> instans atau daftar instans, "
            f"diterima {instances!r}")
    if isinstance(instances, Mapping):
        return dict(instances)
    try:
        items = list(instances)
    except TypeError:
        raise ValueError(
            "instances harus berupa kamus nama -> instans atau daftar instans, "
            f"diterima {type(instances).__name__}") from None
    return {getattr(item, "name", str(k)): item for k, item in enumerate(items)}


def reject_test_instances(instances: dict, allow_test_instances: bool) -> None:
    """Tolak instans uji Brandimarte (MK03, MK04, MK09) untuk pelatihan.

    Instans uji hanya dibuka oleh evaluasi. Pelatihan yang menyentuhnya membuat hasil
    uji tidak lagi bermakna, jadi pemeriksaan ini sengaja keras.

    Parameters
    ----------
    instances : dict
        Hasil ``check_instances``.
    allow_test_instances : bool
        Hanya untuk tes unit. Pipeline tidak pernah mengisi True.

    Raises
    ------
    ValueError
        Bila ada instans uji dan ``allow_test_instances`` bukan True.
    """
    if allow_test_instances is True:
        return
    names = [str(name) for name in instances]
    names += [str(getattr(inst, "name", "")) for inst in instances.values()]
    found = sorted({n for n in names if n in BRANDIMARTE_TEST})
    if found:
        raise ValueError(
            f"fit menolak instans uji {found}. Instans uji tidak boleh dipakai melatih "
            "(pembagian ada di configs/split.yaml).")


def base_seed(random_state) -> int:
    """Seed bilangan bulat dasar dari ``random_state``.

    Bilangan bulat dipakai apa adanya. ``None`` dan ``Generator`` diubah menjadi satu
    bilangan acak. Seed pekerja, penjadwal kapasitas, dan model diturunkan dari angka ini.

    Examples
    --------
    >>> from dfjsp.selectors._common import base_seed
    >>> base_seed(7)
    7
    """
    from ..utils import check_random_state

    if isinstance(random_state, (int, np.integer)) and not isinstance(random_state, bool):
        if random_state < 0:
            raise ValueError(f"random_state tidak boleh negatif, diterima {random_state!r}")
        return int(random_state)
    return int(check_random_state(random_state).integers(0, 2**31 - 1))


# ------------------------------------------------------------ pemeriksaan parameter
def is_number(value) -> bool:
    """Bilangan hingga (bukan bool, bukan NaN, bukan tidak terhingga)."""
    return isinstance(value, (int, float, np.integer, np.floating)) \
        and not isinstance(value, bool) and bool(np.isfinite(value))


def is_whole(value, low: int) -> bool:
    """Bilangan bulat (bukan bool) yang >= ``low``."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) \
        and value >= low


def is_positive_list(value) -> bool:
    """Daftar tidak kosong berisi bilangan > 0."""
    try:
        items = list(value)
    except TypeError:
        return False
    return bool(items) and all(is_number(x) and x > 0 for x in items)


def need(obj, ok: bool, name: str, rule: str) -> None:
    """Lempar ``ValueError`` yang menyebut parameter ``name`` bila ``ok`` salah."""
    if not ok:
        raise ValueError(f"{name} {rule}, diterima {getattr(obj, name)!r}")
