"""A project typed in by a user (epics, issues, developers) as a thesis instance (seam B4).

The project is written in the Brandimarte JSON schema (one job per epic, one operation per
issue, one machine per developer) and then goes through exactly the path of the thesis
instances in ``instans_acak.load_random``: ``parse_brandimarte_json_text``,
``validate_instance``, and ``to_software_project`` with ``k_tight`` 1.5, capacity multiplier
1.2 and domain seed 0. Two things differ, both stated to the user:

* issues inside one epic follow each other (as operations inside a job), and the only
  dependencies between epics are the links the user typed in, so ``p_cross_dep`` is 0 instead
  of the random 0.10 of the thesis instances;
* the policy was trained on seven Brandimarte instances, so a typed-in project is outside
  its training data.

Due dates and the initial schedule are not set here; ``LadderRepairEnv.reset`` sets them as
for every instance (completion in S0 plus half of the effort).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from types import SimpleNamespace


@dataclass(frozen=True)
class CustomProject:
    """A checked user project; build the instance with :func:`build_instance`."""

    key: str
    name: str
    developers: tuple[str, ...]
    epics: tuple[tuple[str, tuple[tuple[str, tuple[tuple[int, float], ...]], ...]], ...]
    links: tuple[tuple[int, int], ...]      # (task id that blocks, task id that waits)

    @property
    def instance_name(self) -> str:
        return f"custom:{self.key}"

    def task_ids(self) -> list[list[int]]:
        out, n = [], 0
        for _, issues in self.epics:
            out.append(list(range(n, n + len(issues))))
            n += len(issues)
        return out


def parse(spec: dict) -> CustomProject:
    """Check the shape of a project and return it; a clear ``ValueError`` names any problem."""
    key = str(spec.get("key", "")).strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9]{1,9}", key):
        raise ValueError("key proyek harus 2-10 huruf/angka, diawali huruf (misalnya TOKO)")
    devs = tuple(str(d.get("name", "")).strip() or f"Pengembang {i + 1}"
                 for i, d in enumerate(spec.get("developers", [])))
    if not devs:
        raise ValueError("proyek butuh paling sedikit satu pengembang")
    epics = []
    for e, epic in enumerate(spec.get("epics", [])):
        issues = []
        for i, issue in enumerate(epic.get("issues", [])):
            where = f"epic {e + 1} issue {i + 1}"
            efforts = []
            for d, h in (issue.get("efforts") or {}).items():
                d, h = int(d), float(h)
                if not 0 <= d < len(devs):
                    raise ValueError(f"{where}: pengembang {d + 1} tidak ada di proyek")
                if not h > 0:
                    raise ValueError(f"{where}: upaya harus lebih dari 0 jam")
                efforts.append((d, h))
            if not efforts:
                raise ValueError(f"{where}: isi upaya untuk paling sedikit satu pengembang")
            issues.append((str(issue.get("summary", "")).strip() or f"Tugas {i + 1}",
                           tuple(sorted(efforts))))
        if not issues:
            raise ValueError(f"epic {e + 1}: butuh paling sedikit satu issue")
        epics.append((str(epic.get("name", "")).strip() or f"Fitur {e + 1:02d}", tuple(issues)))
    if not epics:
        raise ValueError("proyek butuh paling sedikit satu epic")
    project = CustomProject(key, str(spec.get("name", key)), devs, tuple(epics), ())
    ids = project.task_ids()
    links = []
    for link in spec.get("links", []):
        (fe, fi), (te, ti) = link["from"], link["to"]
        try:
            links.append((ids[fe][fi], ids[te][ti]))
        except IndexError:
            raise ValueError(f"tautan {link} menunjuk issue yang tidak ada") from None
    return CustomProject(key, project.name, devs, tuple(epics), tuple(links))


def to_brandimarte(project: CustomProject) -> dict:
    """The project in the JSON schema of ``data/raw/MK*.json``."""
    return {"machines": len(project.developers),
            "jobs": [[[{"machine": d, "processing": h} for d, h in efforts]
                      for _, efforts in issues] for _, issues in project.epics]}


def build_instance(project: CustomProject):
    """Instance of the project through the same path as the thesis instances."""
    import numpy as np
    from dfjsp.core.domain import assert_dag_acyclic, to_software_project
    from dfjsp.core.parser import parse_brandimarte_json_text
    from dfjsp.core.validate import validate_instance

    base = parse_brandimarte_json_text(json.dumps(to_brandimarte(project)),
                                       project.instance_name)
    problems = [p for p in validate_instance(base) if "Tabel 3.2" not in p]
    if problems:
        raise ValueError(f"proyek tidak lolos pemeriksaan: {problems}")
    cfg = SimpleNamespace(k_tight=1.5,
                          domain=SimpleNamespace(capacity_mult=1.2, p_cross_dep=0.0, seed=0),
                          priority=SimpleNamespace(invert=False))
    inst = to_software_project(base, cfg, np.random.default_rng(0))
    by_id = {t.task_id: t for t in inst.tasks}
    for src, dst in project.links:
        if src not in by_id[dst].deps:
            by_id[dst].deps.append(src)
    try:
        assert_dag_acyclic(inst)
    except Exception as exc:                      # noqa: BLE001  (pesan dari pustaka)
        raise ValueError(f"tautan antar-issue membentuk siklus: {exc}") from None
    return inst


def labels(project: CustomProject) -> dict:
    """Names for the board: tasks, developers, epics."""
    ids = project.task_ids()
    return {"tasks": {tid: summary for (_, issues), row in zip(project.epics, ids)
                      for (summary, _), tid in zip(issues, row)},
            "devs": dict(enumerate(project.developers)),
            "epics": {j: name for j, (name, _) in enumerate(project.epics)}}
