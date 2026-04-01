#include "GHAModel.hh"
#include "HDF5Util.hh"
#include <string>
#include <vector>

void GHAModel::WriteNullModelHDF5(string filename){
  cout << "Writing null model result to " << filename << endl;
  H5File* outfile = new H5File(filename.c_str(), H5F_ACC_TRUNC);
  WriteNullModelHDF5(outfile);
  delete outfile;
}


void GHAModel::WriteNullModelHDF5(H5File* outfile){
  // write scalars
  WriteHDF5scalar<double>(outfile, GetEpsilon(),       "epsilon");
  WriteHDF5scalar<double>(outfile, GetTau(),           "tau");
  WriteHDF5scalar<double>(outfile, GetTauLimit(),      "taulimit");
  WriteHDF5scalar<size_t>(outfile, GetFitAP_ns(),      "ns_fit_ap");
  WriteHDF5scalar<size_t>(outfile, GetFitIn_ns(),      "ns_fit_in");
  WriteHDF5scalar<size_t>(outfile, GetNIterationsAP(), "niterations_ap");
  WriteHDF5scalar<size_t>(outfile, GetNIterationsIn(), "niterations_in");
  WriteHDF5scalar<bool>(outfile,   GetBLast(),         "b_last");
  WriteHDF5scalar<bool>(outfile,   GetBFullNR(),       "b_fullnr");
  WriteHDF5scalar<bool>(outfile,   GetConvergedAP(),   "converged_ap");
  WriteHDF5scalar<bool>(outfile,   GetConvergedIn(),   "converged_in");
  // write matrices
  beta0->WriteHDF5(*outfile);
  beta1->WriteHDF5(*outfile);
  lambda->WriteHDF5(*outfile);
  lambda_in->WriteHDF5(*outfile);
  clambda->WriteHDF5(*outfile);
  clambda_in->WriteHDF5(*outfile);
  b_in->WriteHDF5(*outfile);
}


void GHAModel::ReadNullModelHDF5(string filename){
  cout << "Reading null model result from " << filename << endl;
  H5File* infile = new H5File(filename.c_str(), H5F_ACC_RDONLY);
  ReadNullModelHDF5(infile);
  delete infile;
}


void GHAModel::ReadNullModelHDF5(H5File* infile){
  SetSeedsAP();
  SetSeedsIn();
  // read scalars
  ReadHDF5scalar<double>(infile, epsilon,        "epsilon");
  ReadHDF5scalar<double>(infile, tau,            "tau");
  ReadHDF5scalar<double>(infile, taulimit,       "taulimit");
  ReadHDF5scalar<size_t>(infile, ns_fit_ap,      "ns_fit_ap");
  ReadHDF5scalar<size_t>(infile, ns_fit_in,      "ns_fit_in");
  ReadHDF5scalar<size_t>(infile, niterations_ap, "niterations_ap");
  ReadHDF5scalar<size_t>(infile, niterations_in, "niterations_in");
  ReadHDF5scalar<bool>(infile, b_last,           "b_last");
  ReadHDF5scalar<bool>(infile, b_fullnr,         "b_fullnr");
  ReadHDF5scalar<bool>(infile, converged_ap,     "converged_ap");
  ReadHDF5scalar<bool>(infile, converged_in,     "converged_in");
  // cleanup matrices
  if(beta0)      delete beta0;      beta0      = NULL;
  if(beta1)      delete beta1;      beta1      = NULL;
  if(lambda)     delete lambda;     lambda     = NULL;
  if(lambda_in)  delete lambda_in;  lambda_in  = NULL;
  if(clambda)    delete clambda;    clambda    = NULL;
  if(clambda_in) delete clambda_in; clambda_in = NULL;
  if(b_in)       delete b_in;       b_in       = NULL;  
  // read matrices
  beta0      = new PMatrix<double>("beta0",      *infile);
  beta1      = new PMatrix<double>("beta1",      *infile);
  lambda     = new PMatrix<double>("lambda",     *infile);
  lambda_in  = new PMatrix<double>("lambda_in",  *infile);
  clambda    = new PMatrix<double>("clambda",    *infile);
  clambda_in = new PMatrix<double>("clambda_in", *infile);
  b_in       = new PMatrix<double>("b_in",       *infile);
}


void GHAModel::WriteScoreInfoHDF5(string filename){
  cout << "Writing score and information to " << filename << endl;
  H5File* outfile = new H5File(filename.c_str(), H5F_ACC_TRUNC);
  WriteScoreInfoHDF5(outfile);
  delete outfile;
}


void GHAModel::WriteScoreInfoHDF5(H5File* outfile){
  WriteNullModelHDF5(outfile);
  // write scalars
  WriteHDF5scalar<size_t>(outfile, ns_si_ap,     "ns_si_ap");
  WriteHDF5scalar<size_t>(outfile, ns_si_in,     "ns_si_in");
  WriteHDF5scalar<double>(outfile, utau,         "utau");
  WriteHDF5scalar<double>(outfile, utau_in,      "utau_in");
  WriteHDF5vector<size_t>(outfile, index_unq,    "index_unq");
  WriteHDF5vector<size_t>(outfile, index_unq_in, "index_unq_in");
  size_t ghdim = gauss_hermite.size();
  WriteHDF5scalar<size_t>(outfile, ghdim,        "ghdim");
  // write matrices
  T_unq->WriteHDF5(*outfile);
  Ubeta->WriteHDF5(*outfile);
  Ubeta_in->WriteHDF5(*outfile);
  h_unq->WriteHDF5(*outfile);
  h_unq_in->WriteHDF5(*outfile);
  Uh_unq->WriteHDF5(*outfile);
  Uh_unq_in->WriteHDF5(*outfile);
  V_unq->WriteHDF5(*outfile,       "", 1024, 1024, 6);
  V_unq_in->WriteHDF5(*outfile,    "", 1024, 1024, 6);
  Vinv_unq->WriteHDF5(*outfile,    "", 1024, 1024, 6);
  Vinv_unq_in->WriteHDF5(*outfile, "", 1024, 1024, 6);
  Asc->WriteHDF5(*outfile);
  ebx->WriteHDF5(*outfile);
  w1q->WriteHDF5(*outfile);
  ebxw1q->WriteHDF5(*outfile);
  w1_in->WriteHDF5(*outfile);
  ebx_in->WriteHDF5(*outfile);
  ebxw1_in->WriteHDF5(*outfile);
  const size_t chunk = 1 << 30;
  Ebb_in->WriteHDF51d(*outfile,  "", chunk, 6);
  Eebb_in->WriteHDF51d(*outfile, "", chunk, 6);
  Ebbeb_in->WriteHDF5(*outfile,  "", chunk/4, chunk/4, 6);
  // write vectors of matrices
  for(size_t i=0; i<GetGHDim(); i++){
    if(Ebb_cbs[i])   Ebb_cbs[i]->WriteHDF51d(*outfile,   "", chunk, 6);
    if(Eebb_cbs[i])  Eebb_cbs[i]->WriteHDF51d(*outfile,  "", chunk, 6);
    if(Ebbbb_cbs[i]) Ebbbb_cbs[i]->WriteHDF51d(*outfile, "", chunk, 6);
    for(size_t j=0; j<Ebbeb_cbs[i].size(); j++)
      if(Ebbeb_cbs[i][j]) Ebbeb_cbs[i][j]->WriteHDF51d(*outfile, "", chunk, 6);
  }
}


void GHAModel::ReadScoreInfoHDF5(string filename){
  cout << "Reading score and information from " << filename << endl;
  H5File* infile = new H5File(filename.c_str(), H5F_ACC_RDONLY);
  ReadScoreInfoHDF5(infile);
  delete infile;
}


void GHAModel::ReadScoreInfoHDF5(H5File* infile){
  ReadNullModelHDF5(infile);
  // read scalars
  ReadHDF5scalar<size_t>(infile, ns_si_ap,      "ns_si_ap");
  ReadHDF5scalar<size_t>(infile, ns_si_in,      "ns_si_in");
  ReadHDF5scalar<double>(infile, utau,          "utau");
  ReadHDF5scalar<double>(infile, utau_in,       "utau_in");
  ReadHDF5vector<size_t>(infile, index_unq,     "index_unq");
  ReadHDF5vector<size_t>(infile, index_unq_in,  "index_unq_in");
  size_t ghdim = 0;
  ReadHDF5scalar<size_t>(infile, ghdim, "ghdim");
  assert(gauss_hermite.size() == ghdim);
  // cleanup matrices
  if(T_unq)       delete T_unq;       T_unq       = NULL;
  if(Ubeta)       delete Ubeta;       Ubeta       = NULL;
  if(Ubeta_in)    delete Ubeta_in;    Ubeta_in    = NULL;
  if(h_unq)       delete h_unq;       h_unq       = NULL;
  if(h_unq_in)    delete h_unq_in;    h_unq_in    = NULL;
  if(Uh_unq)      delete Uh_unq;      Uh_unq      = NULL;
  if(Uh_unq_in)   delete Uh_unq_in;   Uh_unq_in   = NULL;
  if(V_unq)       delete V_unq;       V_unq       = NULL;
  if(V_unq_in)    delete V_unq_in;    V_unq_in    = NULL;
  if(Vinv_unq)    delete Vinv_unq;    Vinv_unq    = NULL;
  if(Vinv_unq_in) delete Vinv_unq_in; Vinv_unq_in = NULL;
  if(Asc)         delete Asc;         Asc         = NULL;
  if(ebx)         delete ebx;         ebx         = NULL;
  if(ebxw1q)      delete ebxw1q;      ebxw1q      = NULL;
  if(w1q)         delete w1q;         w1q         = NULL;
  if(ebx_in)      delete ebx_in;      ebx_in      = NULL;
  if(ebxw1_in)    delete ebxw1_in;    ebxw1_in    = NULL;
  if(w1_in)       delete w1_in;       w1_in       = NULL;
  if(Ebb_in)      delete Ebb_in;      Ebb_in      = NULL;
  if(Eebb_in)     delete Eebb_in;     Eebb_in     = NULL;
  if(Ebbeb_in)    delete Ebbeb_in;    Ebbeb_in    = NULL;
  // read matrices
  T_unq       = new PMatrix<double>("T_unq",       *infile);
  Ubeta       = new PMatrix<double>("Ubeta",       *infile);
  Ubeta_in    = new PMatrix<double>("Ubeta_in",    *infile);
  h_unq       = new PMatrix<double>("h_unq",       *infile);
  h_unq_in    = new PMatrix<double>("h_unq_in",    *infile);
  Uh_unq      = new PMatrix<double>("h_unq",       *infile);
  Uh_unq_in   = new PMatrix<double>("Uh_unq_in",   *infile);
  V_unq       = new PMatrix<double>("V_unq",       *infile);
  V_unq_in    = new PMatrix<double>("V_unq_in",    *infile);
  Vinv_unq    = new PMatrix<double>("Vinv_unq",    *infile);
  Vinv_unq_in = new PMatrix<double>("Vinv_unq_in", *infile);
  Asc         = new PMatrix<double>("Asc",         *infile);
  ebx         = new PMatrix<double>("ebx",         *infile);
  ebxw1q      = new PMatrix<double>("ebxw1q",      *infile);
  w1q         = new PMatrix<double>("w1q",         *infile);
  ebx_in      = new PMatrix<double>("ebx_in",      *infile);
  ebxw1_in    = new PMatrix<double>("ebxw1_in",    *infile);
  w1_in       = new PMatrix<double>("w1_in",       *infile);
  Ebb_in      = new BDMatrix<double>("Ebb_in", "Ebb_in",
				     *infile, false, false);
  Eebb_in     = new BDMatrix<double>("Eebb_in", "Eebb_in",
				     *infile, false, false);
  Ebbeb_in    = new PMatrix<double>("Ebbeb_in", *infile);
  // cleanup vectors of matrices
  for(size_t i=0; i<ghdim; i++){
    if(Ebb_cbs[i])   delete Ebb_cbs[i];   Ebb_cbs[i]   = NULL;
    if(Eebb_cbs[i])  delete Eebb_cbs[i];  Eebb_cbs[i]  = NULL;
    if(Ebbbb_cbs[i]) delete Ebbbb_cbs[i]; Ebbbb_cbs[i] = NULL;
    for(size_t j=0; j<i; j++)
      if(j < Ebbeb_cbs[i].size()){
	if(Ebbeb_cbs[i][j]) delete Ebbeb_cbs[i][j]; Ebbeb_cbs[i][j] = NULL;
      }
  }
  // read matrices into vectors
  for(size_t i=ghdim; i>0; i--){
    size_t k = ghdim - i;
    string s = "Ebb_cbs_" + to_string(i);
    if(H5Lexists(infile->getId(), (s+"_dim").c_str(),  H5P_DEFAULT) &&
       H5Lexists(infile->getId(), (s+"_data").c_str(), H5P_DEFAULT))
      Ebb_cbs[k]   = new BDMatrix<double>(s, s, *infile, false, false);
    s = "Eebb_cbs_" + to_string(i);
    if(H5Lexists(infile->getId(), (s+"_dim").c_str(),  H5P_DEFAULT) &&
       H5Lexists(infile->getId(), (s+"_data").c_str(), H5P_DEFAULT))
      Eebb_cbs[k]  = new BDMatrix<double>(s, s, *infile, false, false);
    s = "Ebbbb_cbs_" + to_string(i);
    if(H5Lexists(infile->getId(), (s+"_dim").c_str(),  H5P_DEFAULT) &&
       H5Lexists(infile->getId(), (s+"_data").c_str(), H5P_DEFAULT))
      Ebbbb_cbs[k] = new BDMatrix<double>(s, s, *infile, false, false);
    for(size_t j=0; j<i; j++){
      s = "Ebbeb_" + to_string(i) + "_" + to_string(j);
      if(H5Lexists(infile->getId(), (s+"_dim").c_str(),  H5P_DEFAULT) &&
	 H5Lexists(infile->getId(), (s+"_data").c_str(), H5P_DEFAULT))
	Ebbeb_cbs[k][j] = new BDMatrix<double>(s, s, *infile, false, false);
    }
  }
}
