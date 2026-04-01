# Table of Contents

- [Introduction](#introduction)
- [Citation](#citation)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [UK Biobank GWAS Results](#uk-biobank-gwas-results)

---

# Introduction

GADO (Genetic Associations with Disease Onset) is a scalable and accurate software tool for genome-wide association analysis of censored survival data in large-scale biobanks. Based on frailty models, GADO performs single-variant association tests for time-to-event outcomes while accounting for relatedness in large cohorts. GADO provides well-controlled type I error and improved power for large-scale survival GWAS analyses.

---

# Citation

If you use GADO in your research, please cite:

*(Add paper citation here once available.)*

---

# Installation

## Dependencies

GADO requires the following dependencies:

- [HDF5 C++](https://docs.hdfgroup.org/archive/support/HDF5/doc1.8/cpplus_RM/index.html)
- [OpenBLAS](https://www.openblas.net/)
- [LAPACKE](https://www.netlib.org/lapack/lapacke.html)
- [R](https://www.r-project.org/)
  - Must be configured with `--enable-R-shlib`
  - The following R packages must be installed:
    - [Rcpp](https://cran.r-project.org/web/packages/Rcpp/index.html)
    - [RInside](https://cran.r-project.org/web/packages/RInside/index.html)
    - [rhdf5](https://www.bioconductor.org/packages/devel/bioc/html/rhdf5.html)
    - [optparse](https://cran.r-project.org/web/packages/optparse/index.html)
    - [Matrix](https://cran.r-project.org/web/packages/Matrix/index.html)
    - [hdf5r](https://hhoeflin.github.io/hdf5r/)
- [ROOT](https://root.cern/)
  - CMake should be configured with:
    - `-Dr=ON`
    - `-DR_DIR=<path-to-R-build>`
    - `-DR_LIBRARY=<path-to-libR.so>`
    - `-DR_Rcpp_LIBRARY=<path-to-Rcpp.so>`
- [BGEN](https://enkre.net/cgi-bin/code/bgen)
- Optional: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

---

## Building the Project

The CMake build expects the following environment variables to be set:

- `RPATH=<path-to-R-build>`
- `ROOTSYS=<path-to-root-build>`  
  (set by sourcing `bin/thisroot.sh` from the ROOT build directory)
- `BGENPATH=<path-to-bgen-source>`

If **HDF5 C++**, **OpenBLAS**, or **LAPACKE** are installed in non-standard locations, the `CMakeLists.txt` file may need to be modified accordingly.

To build the project:
```
# Clone repository
git clone https://github.com/dlin-lab/GADO

# Create build directory
mkdir GADO_build
cd GADO_build

# Configure (set USE_CUDA=ON for GPU version, OFF for CPU version)
cmake -DUSE_CUDA=<OFF|ON> ../GADO

# Compile
make -j4
```

Setting `USE_CUDA=OFF` runs GADO in CPU mode. To enable GPU acceleration, set `USE_CUDA=ON` if the CUDA toolkit and `nvcc` are available on your system. With GPU support enabled, the null model fitting step can be executed on the GPU.

We provide example environment setup scripts for both CPU and GPU versions of the code: `env.sh` and `env_gpu.sh`, located in `/GADO/env`.

Before running GADO, please edit these bash scripts to specify the correct paths to your dependencies. Once configured, you can initialize the environment by running:

```
source /path/to/env/env.sh        # CPU mode
source /path/to/env/env_gpu.sh    # GPU mode
```
---

# Usage Guide

To improve computational efficiency, **GADO accepts genotype data and GRM matrices in HDF5 format**. We therefore provide scripts to convert commonly used input files into HDF5.

For score statistic variance estimation, GADO provides a **full-calculation version**, which computes the variance directly without using the variance-ratio approximation. This full calculation is used during **variance-ratio estimation**. After the variance ratio is estimated, GADO computes the score statistic variance using the **variance-ratio approximation** and performs the final score tests.

---
## GPU Version of GADO

We provide a GPU implementation of GADO that accelerates **null model fitting** and **score/information calculations**. The usage is the same as the CPU version of GADO, except that the conditional expectation calculations are performed on the GPU.

Using an H100 GPU, GADO accelerates the computation by approximately **2.4--3.0-fold** speedups compared with the CPU implementation.

---

## Important Note on Sample Ordering

GADO assumes that the participant ordering is **identical across all input files**, including:

- the GRM file  
- the covariate and phenotype CSV file  
- the genotype file

Users must ensure that participants appear in the **exact same order** in all input files before running the analysis. Incorrect ordering will lead to **invalid association results**. We provide an example simulated dataset in `/GADO/data` to help users test and understand the functionality of the code below.

---

## Sparse GRM Generation

A sparse GRM can be computed using **FastSparseGRM**.

Detailed instructions are available at:

https://github.com/rounakdey/FastSparseGRM

This step generates a **block-diagonal sparse GRM stored in RData format**, which can then be converted to HDF5 in the next step and used by GADO.

---

## Convert GRM from RData to HDF5

If the user uses `GRM.RData` generated by **FastSparseGRM**, it can be converted to HDF5 using the following scripts.

---

### Small GRM (dgCMatrix format)

If the GRM has a relatively small dimension and is stored as a `dgCMatrix` object in an `.RData` file, convert it to an HDF5 **dense matrix** using:

```bash
Rscript /path/to/GADO/script/R_to_hdf5.R \
-i /path/to/data/GRM.RData \
-o /path/to/data/GRM.hdf5 \
-m sGRM
```

Here, `sGRM` is the object name of the GRM matrix stored in `GRM.RData`.

In this case, the `-M` option is **not required** when running GADO. The matrix will be block-diagonalized automatically when it is loaded into GADO.

---

### Large GRM (dgCMatrix format)

For large GRMs (e.g., biobank-scale datasets) stored as a `dgCMatrix` object in block-diagonal format, convert it to an HDF5 **block-diagonal matrix** using:

```bash
Rscript /path/to/GADO/script/ukb_convert.R \
-i /path/to/large_GRM.RData \
-o /path/to/large_GRM.hdf5 \
-m sGRM
```

Again, `sGRM` is the object name of the GRM matrix in `GRM.RData`.

The block-diagonal structure is preserved when converting to HDF5. Each block of the original matrix will be further block-diagonalized internally by GADO if necessary.

For large datasets, the `-M` option **must be enabled when running GADO**, since the GRM is stored in block-diagonal format. This option is typically required for **biobank-scale analyses**.

---

## Convert Genotype Data from BGEN to HDF5

To accelerate genotype data loading, convert genotype data from **BGEN format to HDF5** using:

```bash
/path/to/GADO_build/bin/bgen_test \
-i /path/to/data/genotype.bgen \
-c 6
```

The `-c` argument specifies the **HDF5 compression level**, and we recommend using **6**.

---

## Key Options in GADO

You can display all available command-line options using:

```bash
/path/to/GADO_build/bin/GADO -h
```

---

### Gauss–Hermite quadrature

```
-g path to GH data
```

Specifies the path to **Gauss–Hermite quadrature data** required by GADO (`/path/to/GADO/gh/data`).  
These files can be generated using the `gauss.hermite` function in R.

```
-G base of GH files
```

Allows users to provide **their own Gauss–Hermite quadrature files**.

```
-n dimension of GH approx
```

Specifies the dimension used for the Gauss–Hermite approximation. We typically use **2**, meaning family blocks with size ≤ 2 are handled using the Gauss–Hermite method.

```
-o order of GH approx
```

Specifies the **number of quadrature points** used in the Gauss–Hermite approximation.

```
-O step GH order
```

Controls the step size when adjusting the Gauss–Hermite quadrature order.

---

### Parallelization

```
-j number of threads
```

Specifies the number of threads used for **parallel computation**.

---

### Variant Reading

```
-l limit number variants to read
```

Limits the number of variants used for **score testing or variance ratio estimation**.

```
-L stop reading variants once r-hat found
```

Stops reading variants once the **variance ratio (r-hat)** has been estimated.

---

### Time Binning

```
-T parameter for rebinning times
```

Specifies the maximum number of **time bins (max_tbins)** to reduce the number of unique observation times, which can speed up computation.

Example:

If observation times span **10 years** and the desired precision is **2 days**, a reasonable setting would be:

```
max_tbins = 3650
```

---

### Null Model

```
-f null model fit output
```

Path for **storing null model fitting results**.

```
-F null model fit input
```

Path for **reading existing null model results**.

---

### Score / Information Matrix

```
-s score/information output
```

Path for **storing score statistics and information matrix results**.

If `-s` is used without previously specifying `-F`, GADO will perform both null model fitting and score/information calculations, allowing these two steps to be completed in a single run with results saved to the path specified by -s.

```
-S score/information input
```

Path for **loading score and information matrix results**.

---

### Score Testing

```
-t score test output
```

Path for **variance ratio estimation or score testing output**.

---

### Variables

```
-c covariate csv file
```

CSV file containing **Ti, delta, and covariates**.

An example format is provided in:

```
/GADO/data
```

Users must follow the **exact column order** in the example file.

```
-C number of covariates
```

Number of covariates in the CSV file.

```
-a adjust mean/sigma of covariates
```

Normalizes **continuous covariates**.

---

### GRM Input

```
-m grm hdf5 file
```

Path to the **GRM HDF5 file**.

```
-M grm block diagonal
```

Indicates that the GRM was generated using **ukb_convert.R** and stored in **block-diagonal format**.

---

### Genotype Input

```
-b bgen to hdf5 variant file
```

Path to the **HDF5 genotype file** converted from BGEN.

```
-B number of variants in csv file
```

Number of variants listed in the input csv file.

---

### Variance Ratio

```
-r use variance ratio for scores
```

Instructs GADO to estimate the **variance ratio** if it is not provided and perform score testing using the approximation.

```
-R set variance ratio (value or file)
```

Specifies either the **variance ratio value** or the path to a file containing it.

---

### Convergence

```
-e set epsilon for null model
```

Convergence threshold for **null model fitting** (default: `1e-4`).

```
-E epsilon for variance ratio
```

Convergence threshold for **variance ratio estimation** (default: `1e-3`).

---

### Saddlepoint Approximation (SPA)

```
-p use SPA when abs(score) >= value
```

Threshold for using **SPA**.  
SPA will be applied when:

```
|score| >= threshold
```

We typically use **2**.

---

### Memory Control

```
-d do not cache Stheta and Igamma
```

Prevents GADO from caching **Stheta and Igamma** during full variance calculation to reduce memory usage (at the cost of increased runtime).

---

### Verbose Mode

```
-v verbose mode
```

Enables **verbose mode**, which prints more detailed runtime information during execution.

---

## GADO Example Usage

Example scripts are provided to illustrate usage with the sample dataset in `/GADO/data`, using GRM.hdf5 as the input. For users working with a large GRM generated via ukb_convert.R, the -M option should be specified. No additional modifications to the code are necessary.

### Example for Null Model Fitting and Score/Information Calculations

```
/path/to/GADO_build/bin/GADO -v -n 2 -C 2 -o 8 \
-g /path/to/GADO/gh/data \
-m /path/to/GADO/data/GRM.hdf5 \
-c /path/to/GADO/data/variables.csv \
-s /path/to/GADO/data/null_model.hdf5 \
-e 1e-4

```

---

### Example for Variance Ratio Estimation

```
/path/to/GADO_build/bin/GADO -v -n 2 -C 2 -o 8 \
-g /path/to/GADO/gh/data \
-m /path/to/GADO/data/GRM.hdf5 \
-c /path/to/GADO/data/variables.csv \
-S /path/to/GADO/data/null_model.hdf5 \
-t /path/to/GADO/data/var_ratio.hdf5 \
-b /path/to/GADO/data/genotype.hdf5 -r -p 2.0 -L
```

---

### Example for Score Testing

```
/path/to/GADO_build/bin/GADO -v -n 2 -C 2 -o 8 \
-g /path/to/GADO/gh/data \
-m /path/to/GADO/data/GRM.hdf5 \
-c /path/to/GADO/data/variables.csv \
-S /path/to/GADO/data/null_model.hdf5 \
-t /path/to/GADO/data/score_test.hdf5 \
-b /path/to/GADO/data/genotype.hdf5 -r -p 2.0 -R /path/to/GADO/data/var_ratio.hdf5
```

---

## Example for Genomic Control

First combine the GWAS results:
```
Rscript /path/to/GADO/script/combine_pval.R -i /path/to/test_list.txt -o /path/to/combined_test.hdf5
```
`test_list.txt` should contain the paths to all GWAS result files. If there is only one GWAS output file from GADO, you can skip this step and directly proceed to the genomic control step using that file.

Next apply genomic control:

```
Rscript /path/to/GADO/script/genomic_control.R -i /path/to/combined_test.hdf5 -o /path/to/combined_test_gc.hdf5
```

The output file `combined_test_gc.hdf5` contains p-values after genomic control correction.

---

# UK Biobank GWAS Results

The full list of significant SNPs and loci identified in the UK Biobank analysis from the paper can be found in /GADO/data.
