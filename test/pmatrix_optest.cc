#include "PMatrix.hh"
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
#include <cblas.h>

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]){
  
  // option handling
  size_t nrows = 0;
  size_t ncols = 0;
  size_t print = 6;
  size_t fprec = 3;
  size_t nthreads = 1;
  bool use_cblas = false;
  static struct option opts[]{
    {"help",          no_argument, NULL, 'h'},
    {"nrows",   required_argument, NULL, 'r'},
    {"print",   required_argument, NULL, 'p'},
    {"fprec",   required_argument, NULL, 'f'},
    {"threads", required_argument, NULL, 't'},
    {"cblas",         no_argument, NULL, 'c'}};
  int opt = getopt_long(argc, argv, "hr:p:f:t:c", opts, NULL);
  while(opt != -1){
    switch(opt){
    case 'h':
      cout << "options:                  " << endl
	   << "  -r number of rows       " << endl
	   << "  -p max row/column print " << endl
	   << "  -f float precision      " << endl
	   << "  -t number of threads    " << endl
	   << "  -c use cblas            " << endl;
      return 0;
    case 'r': nrows =    atoi(optarg); break;
    case 'p': print =    atoi(optarg); break;
    case 'f': fprec =    atoi(optarg); break;
    case 't': nthreads = atoi(optarg); break;
    case 'c': use_cblas = true; break; 
    default : return 1;
    }
    opt = getopt_long(argc, argv, "hr:p:f:t:c", opts, NULL);
  }
  assert(nrows > 0);
  ncols = nrows;

  map<string,
      pair<high_resolution_clock::time_point,
	   high_resolution_clock::time_point> > times;

  times["alloc"].first = high_resolution_clock::now();
  PMatrix<double>* a = new PMatrix<double>("a", nrows, ncols, false);
  times["alloc"].second = high_resolution_clock::now();
  if(print){ cout << "zero matrix: "; a->Print(print, fprec); }
  a->SetUseCblas(use_cblas);
  a->SetNThreads(nthreads);

  times["ident"].first = high_resolution_clock::now();
  a->AddIdentity();
  times["ident"].second = high_resolution_clock::now();
  if(print){ cout << "identity matrix: "; a->Print(print, fprec); }

  PMatrix<double>* b = new PMatrix<double>("b", nrows, ncols, false);
  b->SetUseCblas(use_cblas);
  b->SetNThreads(nthreads);
  times["rng"].first = high_resolution_clock::now();
  b->Random();
  times["rng"].second = high_resolution_clock::now();
  if(print){ cout << "random matrix: "; b->Print(print, fprec); }

  times["oopadd"].first = high_resolution_clock::now();
  PMatrix<double>* c = *a + *b;
  times["oopadd"].second = high_resolution_clock::now();
  if(print){ cout << "ident + random: "; c->Print(print, fprec); }

  times["inpsub"].first = high_resolution_clock::now();
  *c -= *b;
  times["inpsub"].second = high_resolution_clock::now();
  if(print){ cout << "(ident + random) - random: "; c->Print(print, fprec); }
  delete c;
  
  times["mult"].first = high_resolution_clock::now();
  PMatrix<double>* d = (*a) * (*b);
  times["mult"].second = high_resolution_clock::now();
  if(print){ cout << "ident * random: "; d->Print(print, fprec); }
  delete a;
  delete d;
  
  times["trans"].first = high_resolution_clock::now();
  b->Transpose();
  times["trans"].second = high_resolution_clock::now();
  if(print){ cout << "random transpose: "; b->Print(print, fprec); }

  /*times["sqrt"].first = high_resolution_clock::now();
  b->MakeSymmetric(true);
  b->AddIdentity();
  b->Print();
  PMatrix<double>* s = b->Sqrt();
  times["sqrt"].second = high_resolution_clock::now();
  if(print){ cout << "sqrt(random transpose): "; s->Print(print, fprec); }

  times["sqrt_sq"].first = high_resolution_clock::now();
  PMatrix<double>* t = (*s) * (*s);
  times["sqrt_sq"].second = high_resolution_clock::now();
  if(print){ cout << "sqrt(random transpose)^2: "; t->Print(print, fprec); }  
  delete s;
  delete t;*/
  
  times["inverse"].first = high_resolution_clock::now();
  PMatrix<double>* e = b->Inverse();
  times["inverse"].second = high_resolution_clock::now();
  if(print){ cout << "random transpose inverse: "; e->Print(print, fprec); }

  PMatrix<double>* f = (*b) * (*e);
  if(print){ cout << "random transpose * inverse: "; f->Print(print, fprec); }
  //delete e;
  //delete f;

  times["bwinverse"].first = high_resolution_clock::now();
  if(nthreads == 0) b->BlockwiseInverse(e);
  else b->BlockwiseInverse(e, nrows/8);
  times["bwinverse"].second = high_resolution_clock::now();
  if(print){ cout << "random transpose bw inverse: "; e->Print(print, fprec); }

  f = (*b) * (*e);
  if(print){ cout << "random transpose * bw inverse: ";f->Print(print, fprec);}
  delete e;
  delete f;
  
  times["outprod"].first = high_resolution_clock::now();
  PMatrix<double>* g = a->OuterProduct(1, std::min((size_t) 2, ncols));
  times["outprod"].second = high_resolution_clock::now();
  if(print){ cout << "outer identity product: "; g->Print(print, fprec); }
  delete a;
  delete g;
  
  times["dealloc"].first = high_resolution_clock::now();
  delete b;
  times["dealloc"].second = high_resolution_clock::now();

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
  if(!use_cblas) nthreads = 0;
  outfile.open("data/pmatrix_"+to_string(nthreads)+
	       "_"+to_string(nrows)+"_"+to_string(ncols)+".txt");
  for(auto const& p : duration){
    cout << setw(8) << p.first << " "
	 << setprecision(3) << p.second/1.0e3 << " us" << endl;
    outfile << p.first << " " << p.second/1.0e3 << endl;
  }
  cout << endl;
  outfile.close();
  
  return 0;
}
