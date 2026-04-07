#!/bin/bash

set -e

echo "========== GADO Full Installation =========="

########################################
# Need to be modified
########################################
INSTALL_DIR=${1:-"/path/to/GADO/thirdParty"}
# if you have 4 cores available for compilation
N_CORES=${N_CORES:-4} 

echo "Install directory: $INSTALL_DIR"
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

########################################
# 1. OpenBLAS (in-source build)
########################################
echo "---- Installing OpenBLAS ----"
mkdir -p openblas_build

cd OpenBLAS
chmod -R +x .

make -j$N_CORES NOFORTRAN=1 NO_LTO=1 USE_OPENMP=1
make PREFIX=$INSTALL_DIR/openblas_build install

cd ..

########################################
# 2. LAPACKE (CMake out-of-source)
########################################
echo "---- Installing LAPACKE ----"

mkdir -p lapack_build lapack_install
cd lapack_build

cmake ../lapack \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/lapack_install \
  -DBUILD_SHARED_LIBS=ON -DLAPACKE=ON

cmake --build . -j$N_CORES --target install

cd ..

########################################
# 3. HDF5 (in-source build)
########################################
echo "---- Installing HDF5 ----"
mkdir -p hdf5_build
cd hdf5
chmod -R +x .
./configure --prefix=$INSTALL_DIR/hdf5_build \
            --enable-cxx \
            --enable-shared

make -j$N_CORES
make install

cd ..

########################################
# 4. Readline
########################################
echo "---- Building Readline ----"
mkdir readline_build && cd readline-7.0
chmod -R +x .
./configure --prefix=$INSTALL_DIR/readline_build
make -j4
make install

export READLINE_DIR=$INSTALL_DIR/readline_build
export CPPFLAGS="-I$READLINE_DIR/include"
export LDFLAGS="-L$READLINE_DIR/lib"
export LD_LIBRARY_PATH="$READLINE_DIR/lib:$LD_LIBRARY_PATH"

cd ..

########################################
# 5. R (in-source build, VERY IMPORTANT)
########################################
echo "---- Installing R ----"
mkdir -p R-4.4.1_build
tar -xzf R-4.4.1.tar.gz
cd R-4.4.1

chmod -R +x .

./configure --prefix=$INSTALL_DIR/R-4.4.1_build \
            --enable-R-shlib \
            --with-readline \
            --with-blas="$INSTALL_DIR/openblas_build/lib/libopenblas.so" \ 
            --with-lapack

make -j$N_CORES
make install

cd ..

########################################
# 6. R packages
########################################
echo "---- Installing R packages ----"

export R_LIBS=$INSTALL_DIR/R-4.4.1_build/lib64/R/library

$INSTALL_DIR/R-4.4.1_build/bin/Rscript -e '
install.packages(c("igraph","RInside","optparse","Matrix"), repos="https://cloud.r-project.org")
'

$INSTALL_DIR/R-4.4.1_build/bin/R CMD INSTALL Rcpp_1.0.10.tar.gz

export PATH=$INSTALL_DIR/hdf5_build/bin:$PATH

$INSTALL_DIR/R-4.4.1_build/bin/Rscript -e '
install.packages("hdf5r", repos="https://cloud.r-project.org")
'

$INSTALL_DIR/R-4.4.1_build/bin/Rscript -e '
if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager", repos="https://cloud.r-project.org");
BiocManager::install("rhdf5", ask=FALSE, update=FALSE)
'

########################################
# 7. BGEN
########################################
echo "---- Building BGEN ----"

cd bgen
chmod -R +x .
./waf configure
./waf

cd ..

########################################
# 8. ROOT
########################################

echo "---- Building ROOT ----"
mkdir root_install root_build && cd root_build

export PATH=$INSTALL_DIR/R-4.4.1_build/bin:$PATH
export LD_LIBRARY_PATH=$INSTALL_DIR/R-4.4.1_build/lib64:$LD_LIBRARY_PATH
export R_LIBS=$INSTALL_DIR/R-4.4.1_build/lib64/R/library
export CPATH=$INSTALL_DIR/R-4.4.1_build/lib64/R/library/Rcpp/include:$INSTALL_DIR/R-4.4.1_build/lib64/R/include:$CPATH
cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/root_install ../root-6.26.14 -Dr=ON -Dbuiltin_glew=ON -Dssl=OFF -Dgsl=ON -DR_DIR=$INSTALL_DIR/R-4.4.1_build

cmake --build . --target install -j$N_CORES

source $INSTALL_DIR/root_install/bin/thisroot.sh

cd ..

########################################
# 9. gsl
########################################
echo "---- Building gsl ----"
mkdir gsl_build 
cd gsl-2.7
chmod -R +x .
./configure --prefix=$INSTALL_DIR/gsl_build
chmod -R +x .
make -j$N_CORES
chmod -R +x .
make install
cd ..


########################################
# DONE
########################################
echo ""
echo "========== Installation Completed =========="
echo ""
echo "Next steps:"
echo "Edit and source ./GADO/env/env.sh"
echo "Then build GADO"
