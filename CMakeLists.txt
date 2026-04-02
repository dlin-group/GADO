cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data)

link_libraries(-lgfortran)


# The following parts may need to be modifie
############################################
## HDF5
include_directories(AFTER /usr/include/hdf5/serial)
link_directories(/usr/lib/x86_64-linux-gnu/hdf5/serial)
link_libraries(libhdf5.so libhdf5_cpp.so)

## R
include_directories(AFTER $ENV{RPATH}/lib64/R/include)
include_directories(AFTER $ENV{RPATH}/lib64/R/library/Rcpp/include)
include_directories(AFTER $ENV{RPATH}/lib64/R/library/RInside/include)
link_directories($ENV{RPATH}/lib64/R/lib)
link_libraries(libR.so libRblas.so libRlapack.so)
link_directories($ENV{RPATH}/lib64/R/library/Rcpp/libs)
link_libraries(Rcpp.so)
link_directories($ENV{RPATH}/lib64/R/library/RInside/lib)
link_libraries(RInside.so)
## ROOT
include_directories(AFTER $ENV{ROOTSYS}/include)
link_directories($ENV{ROOTSYS}/lib)
link_libraries(-lCore -lImt -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lROOTVecOps -lTree -lTreePlayer -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lMathMore -lThread -lMultiProc -lROOTDataFrame -lRInterface -pthread -lm -ldl -rdynamic)
## BGEN
include_directories(AFTER $ENV{BGENPATH}/genfile/include)
include_directories(AFTER $ENV{BGENPATH}/db/include)
include_directories(AFTER $ENV{BGENPATH}/3rd_party/sqlite3)
link_directories($ENV{BGENPATH}/build $ENV{BGENPATH}/build/db $ENV{BGENPATH}/build/3rd_party/sqlite3)
link_libraries(libbgen.a libdb.a libsqlite3.a)
## OpenBLAS
include_directories(AFTER $ENV{OPENBLASPATH}/include/openblas)
link_libraries($ENV{OPENBLASPATH}/lib64/libopenblas.so)
## LAPACKE
include_directories($ENV{LAPACKEPATH}/include)
link_libraries($ENV{LAPACKEPATH}/lib64/liblapacke.so)
############################################

link_directories(/usr/lib/x86_64-linux-gnu)
link_libraries(libopenblas.so liblapacke.so libzstd.so libz.so)

add_compile_options(-std=c++17 -Wall -Wno-overloaded-virtual -Wno-misleading-indentation -O3)

option(USE_CUDA "use cuda if available" OFF)

if(${USE_CUDA})
	project(grmfit LANGUAGES CXX CUDA)
	include_directories(AFTER SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	link_directories(${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
	link_libraries(-lcublas -lcudart -lnvidia-ml)
	set(USE_CUDA_LIB 1)
else()
	project(grmfit LANGUAGES CXX)
	set(USE_CUDA_LIB 0)
endif()

configure_file(${CMAKE_SOURCE_DIR}/obj/config_defs.hh.in ${CMAKE_SOURCE_DIR}/obj/config_defs.hh)

file(GLOB_RECURSE ios ${CMAKE_SOURCE_DIR}/io/*.c*)
add_library(grmfit_io SHARED ${ios})
if(${USE_CUDA})
	file(GLOB_RECURSE objs ${CMAKE_SOURCE_DIR}/*obj/*.c*)
	add_library(grmfit_obj SHARED ${objs})
	include_directories(AFTER ${CMAKE_SOURCE_DIR}/gobj)
else()
	file(GLOB_RECURSE objs ${CMAKE_SOURCE_DIR}/obj/*.c*)
	add_library(grmfit_obj SHARED ${objs})
endif()

include_directories(AFTER ${CMAKE_SOURCE_DIR}/obj)
include_directories(AFTER ${CMAKE_SOURCE_DIR}/io)
link_directories(${CMAKE_BINARY_DIR}/lib)
link_libraries(grmfit_io grmfit_obj)

add_executable(root_optest     test/root_optest.cc)
add_executable(pmatrix_optest  test/pmatrix_optest.cc)
add_executable(bmatrix_optest  test/bmatrix_optest.cc)
add_executable(bdmatrix_optest test/bdmatrix_optest.cc)
add_executable(pmatrix_iotest  test/pmatrix_iotest.cc)
add_executable(bgen_test       test/bgen_test.cc)

if(${USE_CUDA})
	set_property(TARGET grmfit_obj PROPERTY CUDA_ARCHITECTURES OFF)
	add_executable(gmatrix_optest  test/gmatrix_optest.cc)
	add_executable(bgmatrix_optest test/bgmatrix_optest.cc)
	add_executable(kernel_test     test/kernel_test.cc)
	add_executable(fit_grm_sample  fit/fit_grm_sample.cc)
	add_executable(fit_grm_sim     fit/fit_grm_sim.cc)
	#add_executable(fit_ghagpu_sim  fit/fit_ghagpu_sim.cc)
endif()

add_executable(fit_gha_val         fit/fit_gha_val.cc)
add_executable(fit_gha_sim         fit/fit_gha_sim.cc)
add_executable(fit_gha_ukbsim      fit/fit_gha_ukbsim.cc)
add_executable(GADO   fit/GADO.cc)
add_executable(fit_gha_rhat        fit/fit_gha_rhat.cc)
add_executable(significance_thresh fit/significance_thresh.cc)
