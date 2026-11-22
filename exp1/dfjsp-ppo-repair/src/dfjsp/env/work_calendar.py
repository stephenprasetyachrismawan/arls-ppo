"""Waktu kerja efektif; interval berhenti menggunakan batas [awal, akhir)."""

import math


def _merged(pauses):
    intervals = []
    for start, end in pauses:
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError('jeda harus berhingga dengan akhir >= awal')
        if end > start:
            intervals.append((start, end))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def working_time(start, end, pauses):
    """Lama bekerja pada [start, end), tidak menghitung jeda dua kali."""
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError('rentang waktu harus berhingga dengan akhir >= awal')
    stopped = sum(max(0.0, min(end, b) - max(start, a))
                  for a, b in _merged(pauses))
    return max(0.0, end - start - stopped)


def finish_after_work(start, effort, pauses):
    """Waktu selesai setelah sejumlah upaya, melanjutkan pekerjaan setelah jeda."""
    if not math.isfinite(start) or not math.isfinite(effort) or effort < 0:
        raise ValueError('waktu harus berhingga dan upaya tidak negatif')
    intervals = _merged(pauses)
    cursor = start
    remaining = effort
    for a, b in intervals:
        if remaining == 0:
            return cursor
        if b <= cursor:
            continue
        available = max(0.0, a - cursor)
        if remaining <= available:
            return cursor + remaining
        remaining -= available
        cursor = b
    return cursor + remaining
