# Installation

MechAudit supports Python 3.10 and newer. It is pure Python, while its required `highspy`, NumPy,
pandas, and SciPy dependencies may use platform-specific binary wheels.

## Install from PyPI

Create an isolated environment and install the package:

```console
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install mechaudit
```

Use `.venv/bin/python` on POSIX or `.venv\Scripts\python.exe` on Windows. The examples below use
an activated environment for readability. If a package-index artifact is not available for the
desired version, install the matching release wheel or a reviewed source checkout instead.

The default install includes the required HiGHS backend. Confirm the installation without
running a solve:

```console
python -c "import mechaudit, highspy; print(mechaudit.__version__)"
```

## Optional xarray input support

NumPy arrays and pandas tables are available in the default installation. Install xarray only
when labeled `DataArray` or `Dataset` inputs are needed:

```console
python -m pip install "mechaudit[xarray]"
```

MechAudit's core import and non-xarray workflows are tested without this extra.

## Install a release wheel

Download the wheel from the matching project release, verify that its version and provenance
match the release notes, and install the local artifact:

```console
python -m pip install path/to/mechaudit-0.1.0-py3-none-any.whl
```

Canonical release coordinates are linked from the
[source repository](https://github.com/studio-strata/mechaudit).

## Contributor installation

An editable installation with all development tools and optional xarray support is:

```console
python -m pip install -e ".[dev,xarray]"
```

The full contributor workflow is in
[CONTRIBUTING.md](https://github.com/studio-strata/mechaudit/blob/main/CONTRIBUTING.md).

## Troubleshooting

### HiGHS cannot be imported

Confirm that the interpreter used to run MechAudit is the same interpreter used for installation:

```console
python -m pip show highspy
python -c "import sys, highspy; print(sys.executable); print(highspy.Highs().version())"
```

If pip attempts a local compiler build, first update pip and check whether `highspy` publishes a
wheel for the selected Python version, operating system, and architecture. Do not silently switch
to an unsupported MechAudit dependency version.

### A solve behaves differently on another machine

Record the MechAudit, Python, NumPy, SciPy, and HiGHS versions together with the model fingerprint,
solver options, tolerance policy, and effective optimal-face delta. See
[Numerical diagnostics](numerical-health.md) and the
[reproducibility methodology](methodology.md#reproducibility-and-serialization-boundaries).
