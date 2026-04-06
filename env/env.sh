########################################
# Need to be modified
########################################
export INSTALL_DIR=${1:-"/proj/dlin_lab/projects/GADO_sw/GADO/thirdParty"}
########################################

export LD_LIBRARY_PATH=$INSTALL_DIR/readline_build/lib:$LD_LIBRARY_PATH

export LD_PRELOAD=/usr/lib64/libtinfo.so.6

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
