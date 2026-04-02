#include "config_defs.hh"
#include "PMatrix.hh"
#include "SparseDosageReader.hh"
#include "ThreadPool.hh"
#include "HDF5Util.hh"
#include <H5Cpp.h>
#include <TMath.h>
#include <assert.h>
#include <getopt.h>
#include <cmath>
#include <iomanip>
#include <utility>
#include <chrono>
#ifdef __CUDA__
#include "gha_kernels.h"
#include <cuda_runtime_api.h>
#include <nvml.h>
#endif

using namespace std;
using namespace std::chrono;

double sigma2     = 0.25;
double X_Xt_inv   = 0.0;
vector<string> varfile;
vector<vector<double> > st;
PMatrix<double>* X = NULL;
vector<PMatrix<double>* > Y;
size_t nind = 0;
#ifdef __CUDA__
double* X_dev  = NULL;
double* Y_dev  = NULL;
nvmlDevice_t device;
#endif

void ProcessVariants(uint32_t index){
  cout << "launching index " << index << " on CPU..." << endl;
  const size_t N = Y.size();
  SparseDosageReader* reader = new SparseDosageReader(varfile[index],
						      vector<size_t>(),
						      (1<<28)/1e9);
  PMatrix<double>* G = new PMatrix<double>("G", 1, nind, false);
  size_t ivar = 0;
  auto t = high_resolution_clock::now();
  while(reader->NextVariant(G)){
    double V = X->InnerRowProduct(0, 0, *G);
    V = (G->InnerRowProduct(0, 0) - V * V * X_Xt_inv) * sigma2;    
    double U = 0.0;
    for(size_t i=0; i<N; i++){
      U = Y[i]->InnerRowProduct(0, 0, *G);
      st[index][i] = std::max(st[index][i], U*U/V);
    }
    ivar ++;
    if(ivar % 1000 == 0){
      cout << "index " << index << ": processed " << ivar << " of "
	   << reader->GetNVariants() << " variants, "
	   << duration_cast<nanoseconds>(high_resolution_clock::now()
					 -t).count()*1e-6/ivar
	   << " ms/var" << endl;
    }
  }
  delete reader;
  delete G;
}


#ifdef __CUDA__
void ProcessVariantsGPU(uint32_t index){
  nvmlUtilization_t utilization;
  nvmlDeviceGetUtilizationRates(device, &utilization);
  if(utilization.gpu > 90){
    ProcessVariants(index);
    return;
  }
  cout << "launching index " << index << " on GPU..." << endl;
  const size_t N = Y.size();
  SparseDosageReader* reader = new SparseDosageReader(varfile[index],
						      vector<size_t>(),
						      (1<<28)/1e9);
  PMatrix<double>* G = new PMatrix<double>("G", 1, nind, false);
  size_t ivar = 0;
  double* G_dev = NULL;
  cudaMalloc((void**) &G_dev, nind*sizeof(double));
  double* st_dev = NULL;
  cudaMalloc((void**) &st_dev, N*sizeof(double));
  cudaMemcpy((void*) st_dev, (void*) &(st[index].front()),
	     N*sizeof(double), cudaMemcpyHostToDevice);
  auto t = high_resolution_clock::now();
  while(reader->NextVariant(G)){
    double V = X->InnerRowProduct(0, 0, *G);
    V = (G->InnerRowProduct(0, 0) - V * V * X_Xt_inv) * sigma2;    
    cudaMemcpy((void*) G_dev, (void*) &(G->GetDataConst().front()),
	       nind*sizeof(double), cudaMemcpyHostToDevice);
    ScoreTestSim(Y_dev, G_dev, st_dev, V, nind, N);
    ivar ++;
    if(ivar % 1000 == 0){
      cout << "index " << index << ": processed " << ivar << " variants, "
	   << duration_cast<nanoseconds>(high_resolution_clock::now()
					 -t).count()*1e-6/ivar
	   << " ms/var" << endl;
    }
  }
  delete reader;
  delete G;
  cudaMemcpy((void*) &(st[index].front()), (void*) st_dev,
	     N*sizeof(double), cudaMemcpyDeviceToHost);
  if(G_dev)  cudaFree(G_dev);
  if(st_dev) cudaFree(st_dev);
}
#endif


int main(int argc, char* argv[]){
  size_t nrep    = 10000;
  size_t threads = 1;
  string outfile = "significance_thresh.hdf5";
  static struct option opts[]{
    {"help",          no_argument, NULL, 'h'},
    {"sigmas",  required_argument, NULL, 's'},
    {"threads", required_argument, NULL, 'j'},
    {"varfile", required_argument, NULL, 'r'},
    {"outfile", required_argument, NULL, 'o'}};
  int opt = getopt_long(argc, argv, "hs:n:j:r:o:", opts, NULL);
  while(opt != -1){
    switch(opt){
    case 'h':
      cout << "options:"                         << endl
	   << "  -s value of sigma"              << endl
	   << "  -n number of replicates"        << endl
	   << "  -j concurrent threads"          << endl
	   << "  -r append to variant file list" << endl
	   << "  -o name of output hdf5 file"    << endl;
      return 0;
    case 's': sigma2  =   stod(optarg); break;
    case 'n': nrep    =   atoi(optarg); break;
    case 'j': threads =   atoi(optarg); break;
    case 'o': outfile = string(optarg); break;
    case 'r': varfile.push_back(string(optarg)); break;
    default:
      cout << "unrecognized option -" << string(""+opt) << endl;
      return 1;
    }
    opt = getopt_long(argc, argv, "hs:n:j:r:o:", opts, NULL);
  }
  assert(nrep > 0);

  auto t = high_resolution_clock::now();
  cout << scientific << setprecision(6) << "checking variant files..." << endl;
  size_t nvar = 0;
  for(const auto& v : varfile){
    cout << v << endl;
    SparseDosageReader* reader = new SparseDosageReader(v, vector<size_t>(),
							(1<<24)/1e9);
    if(nind == 0) nind = reader->GetNInd();
    else assert(reader->GetNInd() == nind);
    nvar += reader->GetNVariants();
    delete reader;
    st.push_back(vector<double>(nrep, -numeric_limits<double>::max()));
  }
  cout << "variant files checked in "
       << duration_cast<nanoseconds>(high_resolution_clock::now()
				     -t).count()*1e-9 << " s" << endl;
  if(nind == 0) return 1;

  t = high_resolution_clock::now();
  cout << "generating random data...";
  X = new PMatrix<double>("X", 1, nind, false);
  X->RandomNormal();
  X_Xt_inv = 1.0 / X->InnerRowProduct(0, 0);
  
  Y.resize(nrep, NULL);
  for(size_t i=0; i<nrep; i++){
    Y[i] = new PMatrix<double>("Y_"+to_string(i), 1, nind, false);
    Y[i]->RandomNormal();
    for(size_t j=0; j<nind; j++)
      Y[i]->SetVal(j, Y[i]->GetVal(j)*sqrt(sigma2));
  }
  cout << duration_cast<nanoseconds>(high_resolution_clock::now()
				     -t).count()*1e-9 << " s" << endl;

  #ifdef __CUDA__
  t = high_resolution_clock::now();
  cout << "allocating gpu arrays...";
  cudaMalloc((void**) &X_dev, nind*sizeof(double));
  cudaMemcpy((void*) X_dev, (void*) &X->GetData().front(),
	     nind*sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**) &Y_dev, nrep*nind*sizeof(double));
  for(size_t i=0; i<nrep; i++)
    cudaMemcpy((void*) (Y_dev+i*nind), (void*) &(Y[i]->GetData().front()),
	       nind*sizeof(double), cudaMemcpyHostToDevice);
  cout << duration_cast<nanoseconds>(high_resolution_clock::now()
				     -t).count()*1e-9 << " s" << endl;
  nvmlInit();
  nvmlDeviceGetHandleByIndex(0, &device);
  #endif

  t = high_resolution_clock::now();
  cout << "processing variants..." << endl;
  if(threads < 2)
    for(size_t i=0; i<varfile.size(); i++) ProcessVariants(i);
  else{
    ThreadPool pool = ThreadPool(threads);
    for(uint32_t i=0; i<(uint32_t)varfile.size(); i++){
      #ifdef __CUDA__
      this_thread::sleep_for(milliseconds(1500));
      pool.enqueue(make_pair(ProcessVariantsGPU, i));
      #else
      pool.enqueue(make_pair(ProcessVariants, i));
      #endif
    }
    pool.join();
  }
  cout << "processed " << nvar << " variants, " << nrep << " replicates in "
       << duration_cast<nanoseconds>(high_resolution_clock::now()
				     -t).count()*1e-9 << " s" << endl;

  delete X;
  for(auto p : Y) if(p) delete p;
  #ifdef __CUDA__
  if(X_dev) cudaFree(X_dev);
  if(Y_dev) cudaFree(Y_dev);
  #endif

  vector<double> pmin(nrep, numeric_limits<double>::max());
  for(size_t i=0; i<st.size(); i++)
    for(size_t j=0; j<nrep; j++)
      pmin[j] = std::min(pmin[j], TMath::Prob(st[i][j], 1));
  sort(pmin.begin(), pmin.end());

  cout << "writing min p-values to " << outfile << endl;
  H5File* hfile = new H5File(outfile.c_str(), H5F_ACC_TRUNC);
  WriteHDF5vector<double>(hfile, pmin, "pmin");
  delete hfile;

  return 0;
}
