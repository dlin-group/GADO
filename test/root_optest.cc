#include <TMatrixD.h>
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

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]){

  // option handling
  size_t nrows = 0;
  size_t ncols = 0;
  size_t print = 7;
  static struct option opts[]{
    {"help",        no_argument, NULL, 'h'},
    {"nrows", required_argument, NULL, 'r'},
    {"ncols", required_argument, NULL, 'c'},
    {"print", required_argument, NULL, 'p'}};
  int opt = getopt_long(argc, argv, "hr:c:p:", opts, NULL);
  while(opt != -1){
    switch(opt){
    case 'h':
      cout << "options:                  " << endl
	   << "  -r number of rows       " << endl
	   << "  -c number of columns    " << endl
	   << "  -p max row/column print " << endl;
      return 0;
    case 'r': nrows = atoi(optarg); break;
    case 'c': ncols = atoi(optarg); break;
    case 'p': print = atoi(optarg); break;
    default : return 1;
    }
    opt = getopt_long(argc, argv, "hr:c:p:", opts, NULL);
  }
  assert(nrows > 0);
  assert(ncols > 0);
  if(nrows > 10 || ncols > 10) print = 0;
  
  map<string,
      pair<high_resolution_clock::time_point,
	   high_resolution_clock::time_point> > times;

  times["alloc"].first = high_resolution_clock::now();
  TMatrixD* a = new TMatrixD(nrows, ncols);
  times["alloc"].second = high_resolution_clock::now();
  if(print){ cout << "zero matrix: "; a->Print(); }

  TMatrixD* I = new TMatrixD(nrows, ncols);
  for(int i=0; i<(int)nrows; i++) (*I)[i][i] = 1.0;
  
  times["ident"].first = high_resolution_clock::now();
  (*a) += (*I);
  times["ident"].second = high_resolution_clock::now();
  if(print){ cout << "identity matrix: "; a->Print(); }
  delete I;
  
  TMatrixD* b = new TMatrixD(nrows, ncols);
  times["rng"].first = high_resolution_clock::now();
  double seed = 0.5;
  b->Randomize(0.0, 1.0, seed);
  times["rng"].second = high_resolution_clock::now();
  if(print){ cout << "random matrix: "; b->Print(); }

  times["oopadd"].first = high_resolution_clock::now();
  TMatrixD c = *a + *b;
  times["oopadd"].second = high_resolution_clock::now();
  if(print){ cout << "ident + random: "; c.Print(); }

  times["inpsub"].first = high_resolution_clock::now();
  c -= *b;
  times["inpsub"].second = high_resolution_clock::now();
  if(print){ cout << "(ident + random) - random: "; c.Print(); }
  
  times["mult"].first = high_resolution_clock::now();
  TMatrixD d = (*a) * (*b);
  times["mult"].second = high_resolution_clock::now();
  if(print){ cout << "ident * random: "; d.Print(); }
  delete a;
  
  times["trans"].first = high_resolution_clock::now();
  b->T();
  times["trans"].second = high_resolution_clock::now();
  if(print){ cout << "random transpose: "; b->Print(); }

  times["inverse"].first = high_resolution_clock::now();
  TMatrixD e = b->Invert();
  times["inverse"].second = high_resolution_clock::now();
  if(print){ cout << "random transpose inverse: "; e.Print(); }

  //TMatrixD f = (*b) * e;
  //if(print){ cout << "random transpose * inverse: "; f.Print(); }
  //delete f;
  
  //times["out prod"].first = high_resolution_clock::now();
  //GMatrix<float>* g = a->OuterProduct(1, std::min((size_t) 2, ncols));
  //times["out prod"].second = high_resolution_clock::now();
  //if(print){ cout << "outer identity product: "; g->Print(print, fprec); }

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
  outfile.open("data/root_"+to_string(nrows)+"_"+to_string(ncols)+".txt");
  for(auto const& p : duration){
    cout << setw(8) << p.first << " "
	 << setprecision(3) << p.second/1.0e3 << " us" << endl;
    outfile << p.first << " " << p.second/1.0e3 << endl;
  }
  cout << endl;
  outfile.close();
  
  return 0;
}
