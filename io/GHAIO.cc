#include "GHAIO.hh"
#include "HDF5Util.hh"
#include "PMatrix.hh"
#include <H5Cpp.h>
#include <iostream>

using namespace std;
using namespace H5;

void GHAWriteBGENScoreTestHDF5(GHAModel* model, string filename,
			       SparseDosageReader* reader,
			       vector<double>& pval,
			       vector<double>& pval_cent,
			       vector<double>& score_stat,
			       vector<double>& score_stat_cent,
			       vector<double>& Ugamma,
			       vector<double>& Ugamma_cent,
			       vector<double>& Ugamma_var,
			       vector<double>& maf,
			       vector<double>& mac,
			       double var_ratio){
  cout << "Writing score test statistics to " << filename << endl;
  H5File* outfile = new H5File(filename.c_str(), H5F_ACC_TRUNC);
  GHAWriteBGENScoreTestHDF5(model, outfile, reader, pval, pval_cent,
			    score_stat, score_stat_cent,
			    Ugamma, Ugamma_cent, Ugamma_var,
			    maf, mac, var_ratio);
  delete outfile;
}


void GHAWriteBGENScoreTestHDF5(GHAModel* model, H5File* outfile,
			       SparseDosageReader* reader,
			       vector<double>& pval,
			       vector<double>& pval_cent,
			       vector<double>& score_stat,
			       vector<double>& score_stat_cent,
			       vector<double>& Ugamma,
			       vector<double>& Ugamma_cent,
			       vector<double>& Ugamma_var,
			       vector<double>& maf,
			       vector<double>& mac,
			       double var_ratio){
  if(reader){
    reader->WriteHeader(outfile);
    WriteHDF5scalar<double>(outfile,
			    reader->GetnsPerVariant(),
			    "read_ns_per_var");
  }
  WriteHDF5scalar<double>(outfile,
			  model->GetScoreTest_ns() / (double) score_stat.size(),
			  "calc_ns_per_var");
  WriteHDF5vector<double>(outfile, pval,            "pval");
  // WriteHDF5vector<double>(outfile, pval_cent,       "pval_cent");
  WriteHDF5vector<double>(outfile, score_stat,      "score_stat");
  // WriteHDF5vector<double>(outfile, score_stat_cent, "score_stat_cent");
  WriteHDF5vector<double>(outfile, Ugamma,          "Ugamma");
  // WriteHDF5vector<double>(outfile, Ugamma_cent,     "Ugamma_cent");
  WriteHDF5vector<double>(outfile, Ugamma_var,      "Ugamma_var");
  WriteHDF5vector<double>(outfile, maf,             "maf");
  WriteHDF5vector<double>(outfile, mac,             "mac");
  WriteHDF5scalar<double>(outfile, var_ratio,       "var_ratio");
  if(model->GetVarRatioEstimates().size() > 0)
    WriteHDF5vector<double>(outfile,
			    model->GetVarRatioEstimates(),
			    "var_ratio_estimate");
}
