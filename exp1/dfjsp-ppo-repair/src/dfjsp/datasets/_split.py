"""Pembagian instans menjadi latih dan uji (keputusan pemilik K2)."""

from __future__ import annotations

from ._instance import ProjectInstance

BRANDIMARTE_TRAIN = ("MK01", "MK02", "MK05", "MK06", "MK07", "MK08", "MK10")
BRANDIMARTE_TEST = ("MK03", "MK04", "MK09")


def split_instances(instances, test=BRANDIMARTE_TEST):
    """Bagi ``instances`` menjadi kamus latih dan kamus uji.

    Uji = instans yang namanya ada di ``test``. Latih = semua instans lain, dengan
    urutan asal. Keduanya tidak pernah tumpang tindih.

    Parameters
    ----------
    instances : dict
        Nama instans ke ``ProjectInstance``, misalnya hasil ``load_brandimarte``.
    test : daftar str, default=BRANDIMARTE_TEST
        Nama instans uji. Bawaannya MK03, MK04, MK09.

    Returns
    -------
    train, test : dict, dict
        Objek instans yang sama dengan masukan (tidak disalin).

    Raises
    ------
    ValueError
        Bila ada nama uji yang dobel, nama uji yang tidak ada di ``instances``,
        atau tidak ada instans tersisa untuk latih.

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte, split_instances
    >>> train, test = split_instances(load_brandimarte())
    >>> sorted(test)
    ['MK03', 'MK04', 'MK09']
    >>> len(train)
    7
    """
    test_names = tuple(test)
    if len(set(test_names)) != len(test_names):
        raise ValueError(f"nama uji dobel (tumpang tindih): {list(test_names)}")
    missing = [n for n in test_names if n not in instances]
    if missing:
        raise ValueError(
            f"instans uji hilang dari instances: {missing}. "
            f"Yang tersedia: {list(instances)}")
    train = {n: v for n, v in instances.items() if n not in test_names}
    if not train:
        raise ValueError("tidak ada instans tersisa untuk latih")
    return train, {n: instances[n] for n in test_names}
