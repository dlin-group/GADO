#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "lapacke" for configuration "Release"
set_property(TARGET lapacke APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(lapacke PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "lapack"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/liblapacke.so.3.12.0"
  IMPORTED_SONAME_RELEASE "liblapacke.so.3"
  )

list(APPEND _cmake_import_check_targets lapacke )
list(APPEND _cmake_import_check_files_for_lapacke "${_IMPORT_PREFIX}/lib64/liblapacke.so.3.12.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
