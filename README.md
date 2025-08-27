# SPECFEMPP-PY

This repository holds Python abstractions to the `SPECFEM++` Python bindings.

This is the very first iteration of the package, and it is still under development and is suspect to change.

## Installation

First, we neeed to create a `conda/mamba` environment:

```bash
conda create -n specfempp python=3.12 obspy
# mamba create -n specfempp python=3.12 obspy
```

You can of course choose the environment name you want. Activate the environement

```bash
conda activate specfempp
# mamba activate specfempp
```

Second, we need to install `SPECFEM++`. Download the repository:

```bash
git clone git@github.com:PrincetonUniversity/SPECFEMPP.git
cd SPECFEMPP
```

and install it using `pip`:

```bash
pip install . \
  -C cmake.define.SPECFEM_ENABLE_SIMD=ON \
  -C cmake.define.Kokkos_ENABLE_ATOMICS_BYPASS=ON \
  -C cmake.define.Kokkos_ARCH_NATIVE=ON \
  -C cmake.define Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON 
```

As you can see here we can set `CMake` options through `scikit-build`. Third and final step is installing `SPECFEM++-PY`. Download the repository:

```bash
cd .. # Go back to the parent directory
git clone git@github.com:PrincetonUniversity/specfempp-py.git
cd specfempp-py
```

and install it using `pip` in editable mode in the :

```bash
pip install -e .
```


## Usage Example

See the `examples` folder for examples. For instance, to run the `fluid-solid-bathymetry` example:

```bash
cd examples/fluid-solid-bathymetry
python run.py
```


