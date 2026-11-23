"""Parser format Brandimarte (Tahap B, Subbab 4.2 rancangan)."""

from __future__ import annotations

import json

from .datatypes import FJSPInstance, Task


class BrandimarteParseError(ValueError):
    """Galat format Brandimarte; pesan menyebut job/operasi bila relevan."""


def _tokenize_line(line: str) -> list[int]:
    """Pecah satu baris menjadi daftar integer; abaikan token kosong."""
    return [int(tok) for tok in line.split()]


def _is_skippable(raw: str) -> bool:
    """Baris kosong atau komentar (diawali '#') dilewati."""
    stripped = raw.strip()
    return stripped == "" or stripped.startswith("#")


def parse_brandimarte_text(text: str, name: str) -> FJSPInstance:
    """Parse isi teks Brandimarte menjadi FJSPInstance netral.

    Raises
    ------
    BrandimarteParseError
        Pada header rusak, n_eligible=0, proc_time<=0, machine_id di luar batas, atau
        token baris tidak habis tepat.
    """
    lines = [ln for ln in text.splitlines() if not _is_skippable(ln)]
    if not lines:
        raise BrandimarteParseError(f"{name}: berkas kosong / tanpa data.")

    # Header: n_jobs n_machines [avg]. Kolom ketiga (rata-rata mesin/operasi) bisa
    # berupa pecahan dan SELALU diabaikan; hanya dua token pertama yang dipakai.
    header_tokens = lines[0].split()
    if len(header_tokens) < 2:
        raise BrandimarteParseError(
            f"{name}: header butuh minimal 2 angka (n_jobs n_machines), "
            f"dapat {header_tokens}."
        )
    n_jobs, n_machines = int(header_tokens[0]), int(header_tokens[1])
    if n_jobs <= 0 or n_machines <= 0:
        raise BrandimarteParseError(
            f"{name}: n_jobs={n_jobs}, n_machines={n_machines} harus positif."
        )

    job_lines = lines[1:]
    if len(job_lines) != n_jobs:
        raise BrandimarteParseError(
            f"{name}: header mendeklarasikan {n_jobs} job tetapi ditemukan "
            f"{len(job_lines)} baris job."
        )

    tasks: list[Task] = []
    task_id = 0
    for job_id, raw in enumerate(job_lines):
        toks = _tokenize_line(raw)
        pos = 0

        def _take(what: str) -> int:
            nonlocal pos
            if pos >= len(toks):
                raise BrandimarteParseError(
                    f"{name}: job {job_id}: token habis saat membaca {what}."
                )
            val = toks[pos]
            pos += 1
            return val

        n_op = _take("n_op")
        if n_op <= 0:
            raise BrandimarteParseError(f"{name}: job {job_id}: n_op={n_op} tidak valid.")

        for op_index in range(n_op):
            n_eligible = _take(f"n_eligible (job {job_id}, op {op_index})")
            if n_eligible <= 0:
                raise BrandimarteParseError(
                    f"{name}: job {job_id}, op {op_index}: n_eligible={n_eligible} "
                    f"tidak bermakna (operasi tanpa mesin)."
                )
            eligible: dict[int, float] = {}
            for _ in range(n_eligible):
                machine_1based = _take(f"machine_id (job {job_id}, op {op_index})")
                proc_time = _take(f"proc_time (job {job_id}, op {op_index})")
                machine_id = machine_1based - 1  # basis-1 -> basis-0
                if not (0 <= machine_id < n_machines):
                    raise BrandimarteParseError(
                        f"{name}: job {job_id}, op {op_index}: machine_id "
                        f"{machine_1based} di luar [1, {n_machines}]."
                    )
                if proc_time <= 0:
                    raise BrandimarteParseError(
                        f"{name}: job {job_id}, op {op_index}: proc_time "
                        f"{proc_time} harus > 0."
                    )
                eligible[machine_id] = float(proc_time)

            tasks.append(
                Task(
                    task_id=task_id,
                    job_id=job_id,
                    op_index=op_index,
                    eligible_devs=eligible,
                )
            )
            task_id += 1

        if pos != len(toks):
            raise BrandimarteParseError(
                f"{name}: job {job_id}: {len(toks) - pos} token sisa di akhir baris "
                f"(format rusak)."
            )

    return FJSPInstance(
        name=name,
        n_jobs=n_jobs,
        n_devs=n_machines,
        tasks=tasks,
        developers=[],
        meta={"source": "brandimarte", "n_tasks": len(tasks)},
    )


def parse_brandimarte_json_text(text: str, name: str) -> FJSPInstance:
    """Parse isi JSON Brandimarte (skema SchedulingLab) menjadi FJSPInstance netral.

    Raises
    ------
    BrandimarteParseError
        Pada skema rusak, operasi tanpa mesin, proc_time<=0, atau machine_id di luar
        batas.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BrandimarteParseError(f"{name}: JSON tidak valid: {exc}.") from exc

    if not isinstance(obj, dict) or "machines" not in obj or "jobs" not in obj:
        raise BrandimarteParseError(
            f"{name}: JSON butuh kunci 'machines' dan 'jobs'."
        )

    n_machines = int(obj["machines"])
    jobs = obj["jobs"]
    if n_machines <= 0:
        raise BrandimarteParseError(f"{name}: machines={n_machines} harus positif.")
    if not isinstance(jobs, list) or not jobs:
        raise BrandimarteParseError(f"{name}: 'jobs' harus daftar tidak kosong.")

    n_jobs = len(jobs)
    tasks: list[Task] = []
    task_id = 0
    for job_id, ops in enumerate(jobs):
        if not isinstance(ops, list) or not ops:
            raise BrandimarteParseError(
                f"{name}: job {job_id} harus daftar operasi tidak kosong."
            )
        for op_index, choices in enumerate(ops):
            if not isinstance(choices, list) or not choices:
                raise BrandimarteParseError(
                    f"{name}: job {job_id}, op {op_index}: tanpa mesin eligible."
                )
            eligible: dict[int, float] = {}
            for choice in choices:
                machine_id = int(choice["machine"])
                proc_time = float(choice["processing"])
                if not (0 <= machine_id < n_machines):
                    raise BrandimarteParseError(
                        f"{name}: job {job_id}, op {op_index}: machine_id "
                        f"{machine_id} di luar [0, {n_machines})."
                    )
                if proc_time <= 0:
                    raise BrandimarteParseError(
                        f"{name}: job {job_id}, op {op_index}: proc_time "
                        f"{proc_time} harus > 0."
                    )
                eligible[machine_id] = proc_time

            tasks.append(
                Task(
                    task_id=task_id,
                    job_id=job_id,
                    op_index=op_index,
                    eligible_devs=eligible,
                )
            )
            task_id += 1

    return FJSPInstance(
        name=name,
        n_jobs=n_jobs,
        n_devs=n_machines,
        tasks=tasks,
        developers=[],
        meta={"source": "brandimarte", "format": "json", "n_tasks": len(tasks)},
    )


def parse_brandimarte(path: str, name: str | None = None) -> FJSPInstance:
    """Baca berkas Brandimarte dari `path` menjadi FJSPInstance netral.

    Parameters
    ----------
    path : str
        Jalur berkas (mis. data/raw/MK01.json).
    name : str, optional
        Nama instans; bawaan dari nama berkas tanpa ekstensi.
    """
    import os

    if name is None:
        name = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.lower().endswith(".json"):
        return parse_brandimarte_json_text(text, name)
    return parse_brandimarte_text(text, name)
