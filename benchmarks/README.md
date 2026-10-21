# Benchmarks

`benchmark_smoke.py` is a reproducible release smoke benchmark for a two-type revenue-design
problem. It reports:

- compilation and certified baseline-solve time;
- fast and full optimal-face analysis time;
- auxiliary solve counts;
- traced peak Python memory;
- analytical objective checks and model fingerprints;
- Python, platform, MechAudit, and HiGHS versions.

Run three repetitions:

```console
python benchmarks/benchmark_smoke.py --repeat 3
```

Write the same deterministic schema to a file:

```console
python benchmarks/benchmark_smoke.py --repeat 3 --json benchmark.json
```

The numerical result is asserted, but timing is never a hard CI gate. Solver builds, BLAS
libraries, hardware, thermal state, and operating systems all affect wall-clock measurements.
Compare timings only after confirming that environments, options, fingerprints, and analytical
outputs match.

This small benchmark does not characterize large-domain scaling or production memory
requirements. Performance comparisons should use representative models on the target hardware.
