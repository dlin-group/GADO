#ifndef __GHAIO__
#define __GHAIO__

#include "GHAModel.hh"
#include "SparseDosageReader.hh"
#include "H5Cpp.h"
#include <string>
#include <vector>

using namespace std;

void GHAWriteBGENScoreTestHDF5(GHAModel* model, string filename,
			       SparseDosageReader* reader,
			       vector<double>& pval,
			       vector<double>& score_stat,
			       vector<double>& Ugamma,
			       vector<double>& Ugamma_var,
			       vector<double>& maf,
			       vector<double>& mac,
			       double var_ratio = 0.0);

void GHAWriteBGENScoreTestHDF5(GHAModel* model, H5::H5File* outfile,
			       SparseDosageReader* reader,
			       vector<double>& pval,
			       vector<double>& score_stat,
			       vector<double>& Ugamma,
			       vector<double>& Ugamma_var,
			       vector<double>& maf,
			       vector<double>& mac,
			       double var_ratio = 0.0);

void GHAWriteBGENScoreTestHDF5gc(GHAModel* model, string filename,
                               SparseDosageReader* reader,
                               vector<double>& pval,
                               vector<double>& pval_gc,
                               vector<double>& score_stat,
                               vector<double>& Ugamma,
                               vector<double>& Ugamma_var,
                               vector<double>& maf,
                               vector<double>& mac,
                               double var_ratio = 0.0);

void GHAWriteBGENScoreTestHDF5gc(GHAModel* model, H5::H5File* outfile,
                               SparseDosageReader* reader,
                               vector<double>& pval,
                               vector<double>& pval_gc,
                               vector<double>& score_stat,
                               vector<double>& Ugamma,
                               vector<double>& Ugamma_var,
                               vector<double>& maf,
                               vector<double>& mac,
                               double var_ratio = 0.0);
#endif
