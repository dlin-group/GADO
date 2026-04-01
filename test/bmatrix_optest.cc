#include "BMatrix.hh"
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
  size_t nrows    = 0;
  size_t ncols    = 0;
  size_t nblrows  = 0;
  size_t nblcols  = 0;
  size_t print    = 6;
  size_t fprec    = 3;
  size_t nthreads = 1;
  static struct option opts[]{
    {"help",          no_argument, NULL, 'h'},
    {"nrows",   required_argument, NULL, 'r'},
    {"blsize",  required_argument, NULL, 'b'},
    {"print",   required_argument, NULL, 'p'},
    {"fprec",   required_argument, NULL, 'f'},
    {"threads", required_argument, NULL, 't'}};
  int opt = getopt_long(argc, argv, "hr:b:p:f:t:", opts, NULL);
  while(opt != -1){
    switch(opt){
    case 'h':
      cout << "options:                  " << endl
	   << "  -r number of rows       " << endl
	   << "  -p max row/column print " << endl
	   << "  -f float precision      " << endl
	   << "  -t number of threads    " << endl
	   << "  -b block size           " << endl;
      return 0;
    case 'r': nrows =    atoi(optarg); break;
    case 'p': print =    atoi(optarg); break;
    case 'f': fprec =    atoi(optarg); break;
    case 't': nthreads = atoi(optarg); break;
    case 'b': nblrows  = atoi(optarg); break;
    default : return 1;
    }
    opt = getopt_long(argc, argv, "hr:b:p:f:t:", opts, NULL);
  }
  assert(nrows > 0);
  ncols   = nrows;
  nblcols = nblrows;

  if(nthreads > 0) openblas_set_num_threads(nthreads);

  string workdir = "/media/tcald/T7 Shield/data";

   map<string,
      pair<high_resolution_clock::time_point,
	   high_resolution_clock::time_point> > times;

  times["alloc"].first = high_resolution_clock::now();
  BMatrix<float>* a =new BMatrix<float>("a", nrows, ncols, nblrows, nblcols,
					  false, workdir+"/a.h5");
  a->SetWorkDir(workdir);
  times["alloc"].second = high_resolution_clock::now();
  if(print){ cout << "zero matrix: "; a->BlPrint(print, fprec, 0, 0); }

  times["ident"].first = high_resolution_clock::now();
  a->AddIdentity();
  times["ident"].second = high_resolution_clock::now();
  if(print){ cout <<"identity matrix: "; a->BlPrint(print, fprec, 0, 0); }
  
  BMatrix<float>* b = new BMatrix<float>("b", nrows, ncols, nblrows, nblcols,
					   false, workdir+"/b.h5");
  b->SetWorkDir(workdir);
  times["rng"].first = high_resolution_clock::now();
  b->Random();
  times["rng"].second = high_resolution_clock::now();
  if(print){ cout << "random matrix: "; b->BlPrint(print, fprec, 0, 0); }

  a->SetWorkDir("data");
  times["oopadd"].first = high_resolution_clock::now();
  BMatrix<float>* c = *a + *b;
  times["oopadd"].second = high_resolution_clock::now();
  if(print){ cout << "ident + random: "; c->BlPrint(print, fprec, 0, 0); }

  times["inpsub"].first = high_resolution_clock::now();
  *c -= *b;
  times["inpsub"].second = high_resolution_clock::now();
  if(print){ cout << "(ident + random) - random: ";c->BlPrint(print,fprec,0,0); }
  delete c;

  times["mult"].first = high_resolution_clock::now();
  BMatrix<float>* d = (*a) * (*b);
  times["mult"].second = high_resolution_clock::now();
  if(print){ cout << "ident * random: "; d->BlPrint(print, fprec, 0, 0); }
  delete a;
  delete d;

  times["inverse"].first = high_resolution_clock::now();
  BMatrix<float>* e = b->Inverse();
  times["inverse"].second = high_resolution_clock::now();
  if(print){ cout << "random transpose inverse: "; e->BlPrint(print,fprec,0,0); }
  
  BMatrix<float>* f = (*b) * (*e);
  if(print){cout << "random transpose * inverse: ";f->BlPrint(print,fprec,0, 0);}
  delete e;
  delete f;

  times["bwinverse"].first = high_resolution_clock::now();
  e = b->BlockwiseInverse();
  times["bwinverse"].second = high_resolution_clock::now();
  if(print){cout << "random transpose bw inverse: ";e->BlPrint(print,fprec,0,0);}

  f = (*b) * (*e);
  if(print){cout<<"random transpose * bw inverse: ";f->BlPrint(print,fprec,0,0);}
  f->BlPrint(print, fprec, 0, 1);
  f->BlPrint(print, fprec, 1, 0);
  f->BlPrint(print, fprec, 1, 1);
  delete e;
  delete f;
  
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
  outfile.open("data/bmatrix_"+to_string(nthreads)+
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
