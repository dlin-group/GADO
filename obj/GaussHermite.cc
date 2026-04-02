#include "GaussHermite.hh"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <functional>
#include <numeric>
#include <cstdint>
#include <assert.h>

using namespace std;

GaussHermite::GaussHermite(string path, string basefn, size_t order){
  ghpath = path;
  ghbase = basefn;
  roots   = ReadFile(path + "/" + basefn + "_" + to_string(order) + "_x.txt");
  weights = ReadFile(path + "/" + basefn + "_" + to_string(order) + "_w.txt");
  roots_sqrt2 = roots;
  for(auto& r : roots_sqrt2) r *= M_SQRT2;
  if(roots.size() != order || weights.size() != order){
    cout << "GaussHermite: error reading order " << order
	 << " found " << roots.size() << " roots, "
	 << weights.size() << " weights" << endl;
    roots.clear();
    weights.clear();
    roots_sqrt2.clear();
  }
}

vector<double> GaussHermite::ReadFile(string fname){
  vector<double> vals;
  ifstream f(fname);
  if(!f.is_open()){
    cout << "GaussHermite: error - cannot open " << fname << endl;
    return vals;
  }
  if(!f.good()){
    cout << "GaussHermite: error - cannot read from " << fname << endl;
    if(f.is_open()) f.close();
    return vals;
  }
  string line;
  double val;
  while(getline(f, line)){
    stringstream ss(line);
    while(ss >> val) vals.push_back(val);
  }
  if(f.is_open()) f.close();
  return vals;
}

GaussHermite2D::GaussHermite2D(string path, string basefn, size_t order) :
  GaussHermite(path, basefn, order){
  weightProd.resize(GetOrder(), vector<double>(GetOrder()));
  for(size_t i=0; i<GetOrder(); i++)
    for(size_t j=0; j<GetOrder(); j++)
      weightProd[i][j] = GetWeight(i) * GetWeight(j);
}

const double GaussHermite2D::GetWeightProd(size_t i, size_t j) const{
  return (i<GetOrder() && j<GetOrder()) ? weightProd[i][j] : 0.0;
}

const double GaussHermite2D::GetRootDotProd(size_t i, size_t j,
					    const double& v0,
					    const double& v1) const{
  return (i<GetOrder() && j<GetOrder()) ?
    v0*GetRoot(i) + v1*GetRoot(j) : 0.0;
}

GaussHermiteND::GaussHermiteND(string path, string basefn,
			       size_t order, size_t dim) :
  GaussHermite(path, basefn, order){
  powers.resize(dim+1, 1);
  for(size_t i=powers.size()-2; i<(size_t)-1; i--){
    assert(SIZE_MAX / powers[i+1] >= GetOrder());
    powers[i] *= powers[i+1] * GetOrder();
  }
  weightProd.resize(powers[0], 1.0);
  for(size_t i=0; i<weightProd.size(); i++)
    for(size_t j=0; j<powers.size()-1; j++)
      weightProd[i] *= weights[(i%powers[j])/powers[j+1]];
}

const double GaussHermiteND::GetWeightProd(const vector<size_t>& i) const{
  if(i.size() != GetDimension()) return 0.0;
  for(const size_t& j : i) if(j >= GetOrder()) return 0.0;
  return weightProd[inner_product(i.begin(), i.end(), powers.begin()+1, 0)];
}

const double GaussHermiteND::GetWeightProd(const size_t& i) const{
  return (i < weightProd.size() ? weightProd[i] : 0.0);
}

const void GaussHermiteND::GetRootPermutation(vector<double>& v,
					      const size_t& i) const{
  v.resize(GetDimension());
  for(size_t j=0; j<powers.size()-1; j++)
    v[j] = roots[(i%powers[j])/powers[j+1]];
}

const void GaussHermiteND::GetRootSqrt2Permutation(vector<double>& v,
						   const size_t& i) const{
  v.resize(GetDimension());
  for(size_t j=0; j<powers.size()-1; j++)
    v[j] = roots_sqrt2[(i%powers[j])/powers[j+1]];
}

const double GaussHermiteND::GetRootDotProd(const vector<size_t>& i,
					    const vector<double>& v) const{
  if(i.size() != GetDimension() || i.size() != v.size()) return 0.0;
  double dotprod = 0.0;
  for(size_t j=0; j<i.size(); j++){
    if(i[j] >= GetOrder()) return 0.0;
    dotprod += roots[i[j]] * v[j];
  }
  return dotprod;
}
  

  
