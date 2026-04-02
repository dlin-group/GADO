#include "PMatrix.hh"
#include "GRMModel.hh"
#include <getopt.h>
#include <assert.h>
#include <cmath>
#include <iomanip>
#include <utility>
#include <chrono>
#include <cblas-openblas.h>

using namespace std;
using namespace std::chrono;

int main(){

  string f0 = "/home/tcald/bios/lin/grm/covariates_delta_Ttilde3.csv";
  string f1 = "/home/tcald/bios/lin/grm/psi3.csv";
  PMatrix<double>*     T = new PMatrix<double>("T", f0,
					       vector<size_t>(1, 1));
  PMatrix<double>* delta = new PMatrix<double>("delta", f0,
					       vector<size_t>(1, 2));
  PMatrix<double>*     X = new PMatrix<double>("X", f0,
					       vector<size_t>({3, 4}));
  PMatrix<double>*   psi = new PMatrix<double>("psi", f1,
					       vector<size_t>(0),
					       vector<size_t>(1, 0));

  /*T->DeleteRows(128, 2000-128);
  delta->DeleteRows(128, 2000-128);
  X->DeleteRows(128, 2000-128);
  psi->DeleteRows(128, 2000-128);
  psi->DeleteCols(128, 2000-128);*/

  GRMModel* model = new GRMModel(T, delta, X, psi);
  model->SetEpsilon(1.0e-5);
  model->SetbPartialLaplace(true);
  model->SetbFromLast(true);
  model->LoadGaussHermite("/home/tcald/bios/hermite/data", "hermite", 20);
  model->SetVerbose(true);
  openblas_set_num_threads(12);
  
  auto t0 = high_resolution_clock::now();
  //model->GridSearchTauGrad(0, 1, 0.01, 4.01, 100);

  //model->Solve(0, 1);

  vector<size_t> bsize(400, 3);
  vector<pair<size_t, size_t> > sample;//400, make_pair(0, 1));
  model->SolveGH2(0, 1, bsize, sample);
  
  auto t1 = high_resolution_clock::now();
  cout << scientific << setprecision(9)
       << duration_cast<nanoseconds>(t1-t0).count()/1.0e9 << endl;
  
  /*vector<double> nll = model->GetTauNLL();
  for(size_t i=0; i<nll.size(); i++)
    cout << scientific << setprecision(6) << nll[i] << "," << endl;
  cout << endl;
  vector<vector<double> > bnll = model->GetBetaNLL();
  for(size_t i=0; i<bnll.size(); i++)
    cout << scientific << setprecision(6) << bnll[i][0] << "," << endl;
  cout << endl;
  for(size_t i=0; i<bnll.size(); i++)
    cout << scientific << setprecision(6) << bnll[i][1] << "," << endl;
  cout << endl;
  */

  /*auto t0 = high_resolution_clock::now();
  model->GradientDescentTau(0, 0, 1, 1.0, 0.1);
  auto t1 = high_resolution_clock::now();
  cout << scientific << setprecision(9)
  << duration_cast<nanoseconds>(t1-t0).count()/1.0e9 << endl;*/
  
  //delete model;
  delete T;
  delete delta;
  delete X;
  delete psi;
  
  return 0;
}
