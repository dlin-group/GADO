#include "GMatrix.hh"
//#include "CUDADevice.hh"
#include <getopt.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <map>
#include <utility>
#include <algorithm>
#include <assert.h>
#include <random>
#include <cublas_v2.h>

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]){

  // option handling
  size_t nrows = 0;
  size_t ncols = 0;
  size_t print = 7;
  size_t fprec = 3;
  static struct option opts[]{
    {"help",        no_argument, NULL, 'h'},
    {"nrows", required_argument, NULL, 'r'},
    {"print", required_argument, NULL, 'p'},
    {"fprec", required_argument, NULL, 'f'}};
  int opt = getopt_long(argc, argv, "hr:p:f:", opts, NULL);
  while(opt != -1){
    switch(opt){
    case 'h':
      cout << "options:                  " << endl
	   << "  -r number of rows       " << endl
	   << "  -p max row/column print " << endl
	   << "  -f float precision      " << endl;
      return 0;
    case 'r': nrows = atoi(optarg); break;
    case 'p': print = atoi(optarg); break;
    case 'f': fprec = atoi(optarg); break;
    default : return 1;
    }
    opt = getopt_long(argc, argv, "hr:p:f:", opts, NULL);
  }
  assert(nrows > 0);
  ncols = nrows;

  map<string,
      pair<high_resolution_clock::time_point,
	   high_resolution_clock::time_point> > times;

  cublasHandle_t handle;
  cublasCreate_v2(&handle);

  times["alloc"].first = high_resolution_clock::now();
  GMatrix<double>* a = new GMatrix<double>("a", nrows, ncols, false, handle);
  a->CopyToDevice();
  times["alloc"].second = high_resolution_clock::now();
  if(print){ cout << "zero matrix: "; a->Print(print, fprec); }

  a->SetNoCopyFromDevice(true);
  a->SetRetainOnDevice(true);
  times["ident"].first = high_resolution_clock::now();
  a->AddIdentity();
  times["ident"].second = high_resolution_clock::now();
  a->CopyToDevice();
  if(print){ cout << "identity matrix: "; a->Print(print, fprec); }

  GMatrix<double>* b = new GMatrix<double>("b", nrows, ncols, false, handle);
  b->SetNoCopyFromDevice(true);
  b->SetRetainOnDevice(true);
  times["rng"].first = high_resolution_clock::now();
  b->Random();
  times["rng"].second = high_resolution_clock::now();
  b->CopyToDevice();
  if(print){ cout << "random matrix: "; b->Print(print, fprec); }
  
  times["oopadd"].first = high_resolution_clock::now();
  GMatrix<double>* c = *a + *b;
  times["oopadd"].second = high_resolution_clock::now();
  c->ReadFromDevice();
  if(print){ cout << "ident + random: "; c->Print(print, fprec); }

  times["inpsub"].first = high_resolution_clock::now();
  *c -= *b;
  times["inpsub"].second = high_resolution_clock::now();
  c->ReadFromDevice();
  if(print){ cout << "(ident + random) - random: "; c->Print(print, fprec); }
  delete c;
  
  times["mult"].first = high_resolution_clock::now();
  GMatrix<double>* d = (*a) * (*b);
  times["mult"].second = high_resolution_clock::now();
  d->ReadFromDevice();
  if(print){ cout << "ident * random: "; d->Print(print, fprec); }
  delete a;
  delete d;

  b->ReadFromDevice();
  times["trans"].first = high_resolution_clock::now();
  b->Transpose();
  times["trans"].second = high_resolution_clock::now();
  if(print){ cout << "random transpose: "; b->Print(print, fprec); }

  times["inverse"].first = high_resolution_clock::now();
  GMatrix<double>* e = b->Inverse();
  times["inverse"].second = high_resolution_clock::now();
  e->ReadFromDevice();
  if(print){ cout << "random transpose inverse: "; e->Print(print, fprec); }
  
  GMatrix<double>* f = (*b) * (*e);
  f->ReadFromDevice();
  if(print){ cout << "random transpose * inverse: "; f->Print(print, fprec); }
  delete e;
  delete f;
  
  times["bwinverse"].first = high_resolution_clock::now();
  e = b->BlockwiseInverse(nrows/8);
  times["bwinverse"].second = high_resolution_clock::now();
  e->ReadFromDevice();
  if(print){ cout << "random transpose bw inverse: "; e->Print(print, fprec);}

  f = (*b) * (*e);
  f->ReadFromDevice();
  if(print){ cout << "random transpose * bw inverse: ";f->Print(print, fprec);}
  delete e;
  delete f;
  
  //times["out prod"].first = high_resolution_clock::now();
  //GMatrix<double>* g = a->OuterProduct(1, std::min((size_t) 2, ncols));
  //times["out prod"].second = high_resolution_clock::now();
  //if(print){ cout << "outer identity product: "; g->Print(print, fprec); }
  //delete a;
  //delete g;
    
  times["dealloc"].first = high_resolution_clock::now();
  delete b;
  times["dealloc"].second = high_resolution_clock::now();

  cublasDestroy_v2(handle);
  
  vector<pair<string, size_t> > duration;
  for(auto const& t : times)
    duration.push_back(make_pair(t.first,
				 duration_cast<nanoseconds>(t.second.second-
							    t.second.first).
				 count()));
  sort(duration.begin(), duration.end(),
       [&](auto d0, auto d1){ return d0.second<d1.second; });

  cout << endl;
  ofstream outfile;
  outfile.open("data/gmatrix_"+to_string(nrows)+"_"+to_string(ncols)+".txt");
  for(auto const& p : duration){
    cout << setw(8) << p.first << " "
	 << setprecision(3) << p.second/1.0e3 << " us" << endl;
    outfile << p.first << " " << p.second/1.0e3 << endl;
  }
  cout << endl;
  outfile.close();
  
  return 0;
}
