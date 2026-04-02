export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# module load cuda/12.4

export CUDA_HOME=/path/to/cuda

if [ -d "$CUDA_HOME" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
    export USE_CUDA=ON
    echo "[INFO] CUDA enabled at ${CUDA_HOME}"
else
    export USE_CUDA=OFF
    echo "[INFO] CUDA not found, using CPU mode"
fi


export LD_LIBRARY_PATH=/path/to/hdf5_build/lib:$LD_LIBRARY_PATH
export CFLAGS="-I/path/to/hdf5_build/include $CFLAGS"

export OPENBLASPATH=/path/to/OpenBLAS_install
export LD_LIBRARY_PATH=$OPENBLASPATH/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I/path/to/OpenBLAS_install/include/openblas $CFLAGS"

export LAPACKEPATH=/path/to/lapack_install
export LD_LIBRARY_PATH=$LAPACKEPATH/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I/path/to/lapack_install/include $CFLAGS"

export RPATH=/path/to/R-4.4.1_build
export RLIBS_USER=$RPATH/lib64/R/library
export PATH=$RPATH/bin:$PATH
export LD_LIBRARY_PATH=$RPATH/lib64/R/lib:$LD_LIBRARY_PATH:$RLIBS_USER/Rcpp/libs:$RLIBS_USER/RInclude/lib:$LD_LIBRARY_PATH

export BGENPATH=/path/to/bgen

source /path/to/root-6.32.06_build/bin/thisroot.sh
