# Benchmarks

MechAudit's benchmark driver has two roles. The default smoke benchmark measures compilation, a
baseline solve, fast/full optimal-face analysis, auxiliary solve counts, and peak traced Python
memory on a small analytical model. The optional size sweep makes the row-wise cost of full-face
analysis visible.

Run it from a source checkout:

```console
python benchmarks/benchmark_smoke.py --repeat 3
```

Use `--json benchmark.json` to retain structured output. Each record includes Python, platform,
MechAudit, HiGHS, repeat count, objective value, model fingerprint, timing summaries, solve
counts, and peak traced memory.

Run the standard symmetric-domain sweep with:

```console
python benchmarks/benchmark_smoke.py --repeat 1 --sweep
```

The default cases are `2x4`, `3x3`, and `3x4`, where `AxT` means `A` agents with `T` types each.
To choose cases or compare process workers:

```console
python benchmarks/benchmark_smoke.py --repeat 1 --sweep \
  --sweep-case 2x4 --sweep-case 3x3 --jobs -1
```

Each sweep record includes profiles, total and classifiable rows, columns, compilation and base
solve times, full-analysis time, solve counts, and the model fingerprint. The `3x4` case can take
about a minute on a laptop; larger cases can take much longer.

Full analysis is row-resolved: it solves one maximum-slack LP for every selected
inequality and a minimum-slack LP for every row proved nonuniversal. Consequently, runtime grows
with both the number of semantic rows and the cost of each auxiliary LP. Restricting
`constraint_families` changes the actual question and the task count; it is not merely a solver
tuning option.

Wall-clock values are descriptive rather than pass/fail gates because solver builds, BLAS
libraries, CPUs, and operating systems differ. The benchmark does assert analytical correctness
and successful independent audits before reporting performance.

For meaningful comparisons:

1. use the same MechAudit commit or release and dependency environment;
2. keep solver options and process thread settings identical;
3. run on an otherwise idle machine;
4. retain every repetition rather than only the fastest result;
5. compare model fingerprints and analytical outputs before comparing timings.

The sweep is descriptive evidence, not a performance guarantee. Its size supports routine
reproduction during development; it does not replace a dedicated large-domain scaling study.

## Illustrative scaling run

The following single-run measurements were recorded on Windows 11, Python 3.13.5, and HiGHS
1.15.1. Automatic mode stays serial below 512 tasks in a batch and uses at most four processes
above that threshold.

| Domain | Rows | Auxiliary solves | Serial full | `n_jobs=-1` full |
| --- | ---: | ---: | ---: | ---: |
| 3 agents x 3 types | 351 | 460 | 6.61 s | 6.85 s (serial path) |
| 3 agents x 4 types | 1,024 | 1,420 | 69.74 s | 48.53 s |

The larger case shows a 1.44x speedup in this run; the smaller case demonstrates why automatic
mode does not spawn processes for modest batches. Treat these values as an example of the
measurement procedure, not as a cross-machine guarantee.
