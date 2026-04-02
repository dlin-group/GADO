export LD_LIBRARY_PATH=/path/to/readline7/lib:$LD_LIBRARY_PATH

export LD_PRELOAD=/usr/lib64/libtinfo.so.6

export LD_LIBRARY_PATH=/path/to/icu/source/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/path/to/gsl/lib:$LD_LIBRARY_PATH

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
