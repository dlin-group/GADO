/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the root of the source code       *
 * distribution tree, or in https://www.hdfgroup.org/licenses.               *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "H5private.h"

H5_GCC_DIAG_OFF("larger-than=")
H5_CLANG_DIAG_OFF("overlength-strings")

/* clang-format off */
const char H5build_settings[]=
    "        SUMMARY OF THE HDF5 CONFIGURATION\n"
    "        =================================\n"
    "\n"
    "General Information:\n"
    "-------------------\n"
    "                   HDF5 Version: 1.14.5\n"
    "                  Configured on: Fri Apr  3 16:48:20 EDT 2026\n"
    "                  Configured by: qiyaoqin@longleaf-login4.its.unc.edu\n"
    "                    Host system: x86_64-pc-linux-gnu\n"
    "              Uname information: Linux longleaf-login4.its.unc.edu 5.14.0-570.51.1.el9_6.x86_64 #1 SMP PREEMPT_DYNAMIC Sat Sep 27 01:10:11 EDT 2025 x86_64 x86_64 x86_64 GNU/Linux\n"
    "                       Byte sex: little-endian\n"
    "             Installation point: /proj/dlin_lab/projects/GADO_sw/GADO/thirdParty/hdf5_build\n"
    "\n"
    "Compiling Options:\n"
    "------------------\n"
    "                     Build Mode: production\n"
    "              Debugging Symbols: no\n"
    "                        Asserts: no\n"
    "                      Profiling: no\n"
    "             Optimization Level: high\n"
    "\n"
    "Linking Options:\n"
    "----------------\n"
    "                      Libraries: static, shared\n"
    "  Statically Linked Executables: \n"
    "                        LDFLAGS: \n"
    "                     H5_LDFLAGS: \n"
    "                     AM_LDFLAGS: \n"
    "                Extra libraries: -lsz -lz -ldl -lm \n"
    "                       Archiver: ar\n"
    "                       AR_FLAGS: cr\n"
    "                         Ranlib: ranlib\n"
    "\n"
    "Languages:\n"
    "----------\n"
    "                              C: yes\n"
    "                     C Compiler: /usr/bin/gcc ( gcc (GCC) 11.5.0 20240719 )\n"
    "                       CPPFLAGS: \n"
    "                    H5_CPPFLAGS: -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L   -DNDEBUG -UH5_DEBUG_API -I/proj/dlin_lab/projects/GADO_sw/GADO/thirdParty/hdf5/src/H5FDsubfiling\n"
    "                    AM_CPPFLAGS: \n"
    "                        C Flags: \n"
    "                     H5 C Flags:  -std=c99  -Wall -Wcast-qual -Wconversion -Wextra -Wfloat-equal -Wformat=2 -Winit-self -Winvalid-pch -Wmissing-include-dirs -Wshadow -Wundef -Wwrite-strings -pedantic -Wno-c++-compat -Wlarger-than=2560 -Wlogical-op -Wframe-larger-than=16384 -Wpacked-bitfield-compat -Wsync-nand -Wno-unsuffixed-float-constants -Wdouble-promotion -Wtrampolines -Wstack-usage=8192 -Wmaybe-uninitialized -Wdate-time -Warray-bounds=2 -Wc99-c11-compat -Wduplicated-cond -Whsa -Wnormalized -Wnull-dereference -Wunused-const-variable -Walloca -Walloc-zero -Wduplicated-branches -Wformat-overflow=2 -Wformat-truncation=1 -Wattribute-alias -Wshift-overflow=2 -Wattribute-alias=2 -Wmissing-profile -Wc11-c2x-compat -fstdarg-opt -fdiagnostics-urls=never -fno-diagnostics-color -s  -Wbad-function-cast -Wcast-align -Wformat -Wimplicit-function-declaration -Wint-to-pointer-cast -Wmissing-declarations -Wmissing-prototypes -Wnested-externs -Wold-style-definition -Wpacked -Wpointer-sign -Wpointer-to-int-cast -Wredundant-decls -Wstrict-prototypes -Wswitch -Wunused-but-set-variable -Wunused-variable -Wunused-function -Wunused-parameter -Wincompatible-pointer-types -Wint-conversion -Wshadow -Wrestrict -Wcast-function-type -Wmaybe-uninitialized -Wcast-align=strict -Wno-aggregate-return -Wno-inline -Wno-missing-format-attribute -Wno-missing-noreturn -Wno-overlength-strings -Wno-jump-misses-init -Wstrict-overflow=2 -Wno-suggest-attribute=const -Wno-suggest-attribute=noreturn -Wno-suggest-attribute=pure -Wno-suggest-attribute=format -Wno-suggest-attribute=cold -Wno-suggest-attribute=malloc -O3\n"
    "                     AM C Flags: \n"
    "               Shared C Library: yes\n"
    "               Static C Library: yes\n"
    "\n"
    "\n"
    "                        Fortran: no\n"
    "\n"
    "                            C++: yes\n"
    "                   C++ Compiler: /usr/bin/g++ ( g++ (GCC) 11.5.0 20240719 )\n"
    "                      C++ Flags: \n"
    "                   H5 C++ Flags:   -std=c++11 -Wall -Wcast-qual -Wconversion -Wctor-dtor-privacy -Weffc++ -Wextra -Wfloat-equal -Wformat=2 -Winit-self -Winvalid-pch -Wmissing-include-dirs -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual -Wreorder -Wshadow -Wsign-promo -Wundef -Wwrite-strings -pedantic -Wlarger-than=2560 -Wlogical-op -Wframe-larger-than=16384 -Wpacked-bitfield-compat -Wsync-nand -Wdouble-promotion -Wtrampolines -Wstack-usage=8192 -Wmaybe-uninitialized -Wdate-time -Wopenmp-simd -Warray-bounds=2 -Wduplicated-cond -Whsa -Wnormalized -Wnull-dereference -Wunused-const-variable -Walloca -Walloc-zero -Wduplicated-branches -Wformat-overflow=2 -Wformat-truncation=1 -Wattribute-alias -Wshift-overflow=2 -Wattribute-alias=2 -Wmissing-profile -Wno-deprecated-copy -fstdarg-opt -fdiagnostics-urls=never -fno-diagnostics-color -s  -Wcast-align -Wmissing-declarations -Wpacked -Wredundant-decls -Wswitch -Wunused-but-set-variable -Wunused-function -Wunused-variable -Wunused-parameter -Wshadow -O3\n"
    "                   AM C++ Flags: \n"
    "             Shared C++ Library: yes\n"
    "             Static C++ Library: yes\n"
    "\n"
    "                           Java: no\n"
    "\n"
    "\n"
    "Features:\n"
    "---------\n"
    "                     Parallel HDF5: no\n"
    "  Parallel Filtered Dataset Writes: no\n"
    "                Large Parallel I/O: no\n"
    "                High-level library: yes\n"
    "Dimension scales w/ new references: no\n"
    "                  Build HDF5 Tests: yes\n"
    "                  Build HDF5 Tools: yes\n"
    "                   Build GIF Tools: no\n"
    "                      Threadsafety: no\n"
    "               Default API mapping: v114\n"
    "    With deprecated public symbols: yes\n"
    "            I/O filters (external): deflate(zlib),szip(encoder)\n"
    "                  _Float16 support: no\n"
    "                     Map (H5M) API: no\n"
    "                        Direct VFD: no\n"
    "                        Mirror VFD: no\n"
    "                     Subfiling VFD: no\n"
    "                (Read-Only) S3 VFD: no\n"
    "              (Read-Only) HDFS VFD: no\n"
    "    Packages w/ extra debug output: none\n"
    "                       API tracing: no\n"
    "              Using memory checker: no\n"
    "                  Use file locking: best-effort\n"
    "         Strict file format checks: no\n"
    "      Optimization instrumentation: no\n"
;
/* clang-format on */

H5_GCC_DIAG_ON("larger-than=")
H5_CLANG_DIAG_OFF("overlength-strings")
