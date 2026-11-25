# ARLS-PPO: Aggregated-state Repair-Level Selector PPO

Code, locked models, data, and results of the master's thesis

**"Mekanisme *Dynamic Rescheduling* Berbasis *Proximal Policy Optimization* dengan Representasi
Keadaan Teragregasi untuk Manajemen Proyek Perangkat Lunak"**
(*A Dynamic Rescheduling Mechanism Based on Proximal Policy Optimization with Aggregated State
Representation for Software Project Management*), Stephen Prasetya Chrismawan, Master of
Computer Science, Universitas Gadjah Mada, 2026.

Every number in Chapter V of the thesis comes from the code and model files in this
repository, recipe **E15 E9c**, locked on 19 November 2026 before the test data was opened.
The interactive simulator built on top of this repository is
[ScheduleSim](https://github.com/stephenprasetyachrismawan/ScheduleSim).

## The method in one paragraph

A software project is modelled as a Dynamic Flexible Job Shop Scheduling Problem (jobs →
features, machines → developers, operations → tasks). Five disruptions are simulated:

- unplanned work, effort revision, and rework after a failed review, calibrated on the TAWOS
  Jira dataset;
- developer unavailability and priority change, which are assumptions.

Disruptions that arrive within one mean task effort form one *event*. Every event is first
absorbed by **AOR** (affected-operation rescheduling, a right-shift that keeps every queue and
assignment, so it is always feasible). Then, for each disruption *d_j* in order of arrival, a
selector picks a **repair level** for the affected set *A_j*:

- 0: keep AOR;
- 1: resequence;
- 2: reassign at most three tasks;
- 3: partial rescheduling of at most eight tasks.

Choices are judged by one combined value

> J = Tard / E + λ · U,  U = tasks moved to another developer + (n − LIS) for the rest,

where λ is the price of one change. The selector is a MaskablePPO policy (MLP 128×128, Tanh)
that reads **38 aggregated numbers**: 9 about the whole schedule and 29 about the current
disruption and *A_j*. Levels without a feasible candidate are masked. A **guard** rejects a
level whose J is worse than keeping the current schedule. The policy is trained by behaviour
cloning from a Sequential Lookahead teacher, plus one DAgger round (86,832 rows), followed by
100,000 PPO steps.

## Main results (test data, from `latex/.../chapter5/generated/bab5_uji_ringkas.json`)

Relative difference in J of ARLS-PPO (mean of three model seeds) against each comparison
method, in percent. Negative means ARLS-PPO is lower (better). Measure 1 is the mean ratio per
condition and measure 2 the ratio of totals; every instance has equal weight. The intervals are
the 98.75% Bonferroni family over 38 test instances (MK03, MK04, MK09 and 35 public FJSP
benchmarks), using a cluster bootstrap.

| comparison | λ | measure 1 | measure 2 | thesis sentence |
|---|---|---|---|---|
| Random + guard | 0.3 | −24.01 [−26.59; −21.38] | −23.62 [−25.58; −21.74] | better |
| Random + guard | 3 | −14.88 [−16.74; −12.99] | −15.73 [−17.10; −14.11] | better |
| Fixed-L3 + guard | 0.3 | +18.14 [+7.86; +32.65] | −4.45 [−6.29; −0.91] | no clear difference |
| Fixed-L3 + guard | 3 | +0.49 [−2.17; +4.20] | −6.31 [−7.62; −4.30] | no clear difference |

Success indicators:

- **I-1:** the p95 selector time of ARLS-PPO is at most 1.92 ms per instance, below 50 ms.
- **I-2:** the independent schedule verifier found 0 hard violations in 154,980 test episodes.

The full analysis is in `revisi/hasil/gerbang_batch/HASIL_UJI.md` (Indonesian).

## Repository layout

The layout mirrors the thesis repository on purpose. The scripts are therefore byte-for-byte
the files that produced the thesis numbers, every relative path inside them still works, and
every path in Appendix 9 of the thesis exists here.

| Path | Content |
|---|---|
| `exp1/dfjsp-ppo-repair/src/dfjsp/` | Scheduling library (environment, TAWOS disruptions, repair operators, verifier, statistics) |
| `exp1/dfjsp-ppo-repair/data/raw/` | Brandimarte MK01-MK10 and the benchmark instances (`SchedulingLab/fjsp-instances`, commit `ac4c340`, MIT) with SHA-256 checksums |
| `exp1/dfjsp-ppo-repair/configs/` | Data split, checksums, TAWOS calibration (`gangguan_tawos.json`) |
| `exp1/dfjsp-ppo-repair/runs/e15_validasi/` | **Locked models** (`e9c_s{0,1,2}/bc_ppo.zip`, `bc.zip`, `norm.pkl`, `hgb7d.pkl`, `b3_model.pkl`), teacher and DAgger data, training logs, `SHA256SUMS` |
| `revisi/hasil/gerbang_batch/` | Experiment scripts of recipe E15 E9c, their tests, validation results (`e15/`), the test opening log (`buka_uji.log`) and analysis |
| `latex/.../chapter5/generated/` | Tables and JSON of Chapter V, written by `bab5_jalan.py` |
| `docs/KUNCI_UJI.md` | Test lock document (Indonesian): recipe, model hashes, methods, measures, claims |
| `src/arls_ppo/` | New thin API: model check (`models`), one episode with an optional trace (`episode`), full validation parity (`parity`), user projects (`custom`), the user as the selector (`manual`) |
| `results/test/test_episodes.csv.gz` | One row per test episode (154,980 rows), without per-decision timing lists |
| `PROVENANCE.sha256` | SHA-256 of the 343 files copied unchanged from the thesis repository at commit `f9f59439` |

The full test episode files (about 560 MB as JSON) are attached to the
[v1.0.0 release](https://github.com/stephenprasetyachrismawan/arls-ppo/releases/tag/v1.0.0) as
`test_episodes_v1.0.0.tar.xz`. Extract them into `revisi/hasil/gerbang_batch/uji/`.

## Installation

Python 3.11 or newer (the thesis used 3.14.4). A CPU is enough.

```bash
pip install -r exp1/dfjsp-ppo-repair/requirements-lock.txt   # exact versions, CPU torch
pip install -e .
sha256sum -c --ignore-missing exp1/dfjsp-ppo-repair/runs/e15_validasi/SHA256SUMS
pytest                                                        # API tests, about 2 minutes
cd revisi/hasil/gerbang_batch && PYTHONPATH=.:../../../exp1/dfjsp-ppo-repair/src pytest
```

## Running one episode

```python
from arls_ppo.episode import run_episode, trace_episode, METHODS

# MK01, capacity 1.5, intensity 0.15, validation seed 5000, ARLS-PPO model seed 0, λ = 3
r = run_episode("MK01", 1.5, 0.15, 5000, "ppo_bc_ppo_g", 3.0, model_seed=0)
print(r["J"], r["telat"], r["ubah"])         # J, Tard/E, number of changes

t = trace_episode("MK01", 1.5, 0.15, 5000, "ppo_bc_ppo_g", 3.0)
ev = t.events[0]
print(ev["disruptions"], [s["requested"] for s in ev["subs"]], ev["subs"][0]["probs"])
```

`METHODS` holds the eight reported methods:

| code | name in the thesis |
|---|---|
| `ppo_bc_ppo_g` | ARLS-PPO |
| `ppo_bc_ppo` | ARLS-PPO without guard |
| `ppo_bc_g` | ARLS-BC |
| `aor` | AOR |
| `seq` | Sequential Lookahead |
| `fix3_g` | Fixed-L3 + guard |
| `acak_g` | Random + guard |
| `il_g` | Boosting-IL + guard |

Both functions call `cek_episode.run`, the same function the test opening used. Before any
model is loaded, `arls_ppo.models.verify_models()` checks it against the hashes in
`docs/KUNCI_UJI.md` and refuses a model that differs.

`trace_episode` only watches the run. Its result equals `run_episode` exactly; this is tested
for all eight methods in `tests/test_trace.py`. On top of the result it records:

- the schedule before the disruptions, after AOR, and after the repair;
- for every sub-decision: the 38 state numbers, the mask, the level probabilities, the level
  requested and the level used, the guard decision, and J before and after.

### The number log (`explain=True`, since v1.1.0)

`trace_episode(..., explain=True)` also writes down where every number comes from, so a reader
can follow one episode from input to output:

- **input** (`t.explain["input"]`): tasks, capacity `cap_j = m × (Σ nominal effort / developers)`,
  S0 (rebuilt from the instance file and compared with the run), due dates
  `d_i = C_i + 0.5 × (C_i − S_i)`, and `E` as a sum over a count;
- **disruptions** (`t.explain["disruptions"]`): the disruption seed from `derive_episode_seed`,
  the TAWOS parameters used, every disruption, and how they were grouped into events. The list
  is drawn a second time from the seed and compared with the run; the seed does not contain the
  method name, so every method meets the same disruptions;
- **every event** (`ev["explain"]`): what each disruption changed (old and new effort, new due
  date, pause), which tasks AOR shifted, and why each task belongs to A_j;
- **every sub-decision** (`sub["explain"]`): the 38 state numbers, each with its formula and
  inputs; the selector (for the PPO actor: normalisation, both Tanh layers, the four logits and
  the masked softmax); J of levels 0-3 written out as late tasks, moved tasks, and per developer
  the old queue, the new queue, the LIS and n − LIS; and the guard decision;
- **output** (`t.explain["output"]`): final tardiness, total changes, and J.

Each explained number is computed at the moment the thesis code computes the same number and is
compared with it; a difference stops the run (`arls_ppo._explain.ExplainMismatch`). The tests in
`tests/test_explain.py` check the log against the plain run, the numbers the selector read,
`score` and `_lis_length`, and the worked example of thesis section 4.6
(`tests/fixtures/kejadian_contoh.json`), number by number. Times in milliseconds are the only
numbers that change between two runs.

Two more modules serve the simulator; they are not used for any thesis number:

- `arls_ppo.custom`: a user project (epics, issues, developers, links) goes through the path of
  the thesis instances;
- `arls_ppo.manual`: the user picks levels, or moves tasks on the board, inside the same episode
  code. The same J and the same schedule checks apply. Choosing ARLS-PPO's own levels with the
  guard reproduces ARLS-PPO's result exactly (`tests/test_manual.py`).

## Reproducing

- **Validation parity:** `python -m arls_ppo.parity --processes 8` reruns all 9,828 stored
  E15 validation episodes. The requirement is a difference of 0.0, the same as section 10 of
  the test lock.
- **Training:** `revisi/hasil/gerbang_batch/e15/jalan_e15.sh` (with `B3` and `OUT` set) runs
  the whole pipeline: teacher data, BC, DAgger, PPO, boosting, and validation.
- **Test opening:** `buka_uji.py --buka-uji --log <new log> ...` (see its docstring). The
  shipped `buka_uji.log` records the one opening of 19 November 2026. A second opening with
  the same log is refused, so pass a new `--log` when reproducing.
- **Chapter V tables:** `bab5_jalan.py uji uji` (after extracting the release files) writes
  the tables and JSON in `latex/.../chapter5/generated/`.

## Test data policy

The test data (MK03, MK04, MK09 with episode seeds 10000-10009, and 35 benchmark instances with
seeds 11000-11004) was opened once, on 19 November 2026, for the recipe in this repository. It
is locked in three places:

- `penjaga_uji` blocks the test instance files in every experiment process;
- `load_benchmark` refuses benchmark test instances unless the official opening allows them;
- `arls_ppo.episode` also refuses the test episode seeds.

MK03, MK04 and MK09 had been opened once before (20 October 2026) for an earlier method
(MPPO-SRS); the thesis states this.

## License and data

- Code: MIT (see `LICENSE`).
- Brandimarte (1993) instances.
- Hurink, Barnes and Dauzère-Pérès instances from
  [SchedulingLab/fjsp-instances](https://github.com/SchedulingLab/fjsp-instances) (MIT).
- The disruption model is calibrated on TAWOS v1.1 (Tawosi et al., MSR 2022,
  doi:10.5522/04/21308124); only the calibrated parameters are included.
