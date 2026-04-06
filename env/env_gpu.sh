########################################
# Need to be modified
########################################
export INSTALL_DIR=${1:-"/path/to/GADO/thirdParty"}
########################################

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


export LD_LIBRARY_PATH=$INSTALL_DIR/gsl_build/lib:$LD_LIBRARY_PATH

export HDF5PATH=$INSTALL_DIR/hdf5_build
export LD_LIBRARY_PATH=$INSTALL_DIR/hdf5_build/lib:$LD_LIBRARY_PATH
export CFLAGS="-I$INSTALL_DIR/hdf5_build/include $CFLAGS"

export OPENBLASPATH=$INSTALL_DIR/openblas_build
export LD_LIBRARY_PATH=$OPENBLASPATH/lib:$LD_LIBRARY_PATH
export CFLAGS="-I$INSTALL_DIR/OpenBLAS_build/include/openblas $CFLAGS"

export LAPACKEPATH=$INSTALL_DIR/lapack_build
export LD_LIBRARY_PATH=$LAPACKEPATH/lib:$LD_LIBRARY_PATH
export CFLAGS="-I$INSTALL_DIR/lapack_build/include $CFLAGS"

export RPATH=$INSTALL_DIR/R-4.4.1_build
export RLIBS_USER=$RPATH/lib64/R/library
export PATH=$RPATH/bin:$PATH
export LD_LIBRARY_PATH=$RPATH/lib64/R/lib:$LD_LIBRARY_PATH:$RLIBS_USER/Rcpp/libs:$RLIBS_USER/RInclude/lib:$LD_LIBRARY_PATH
export CPATH=$INSTALL_DIR/R-4.4.1_build/lib64/R/library/Rcpp/include:$INSTALL_DIR/R-4.4.1_build/lib64/R/include:$CPATH

export BGENPATH=$INSTALL_DIR/bgen

source $INSTALL_DIR/root_install/bin/thisroot.sh
