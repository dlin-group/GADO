/* read files and provide access for gauss-hermite roots and weights from:
   https://people.sc.fsu.edu/~jburkardt/cpp_src/hermite_rule/hermite_rule.html
*/
#ifndef __GaussHermite__
#define __GaussHermite__

#include <string>
#include <vector>

using namespace std;

class GaussHermite{
public:
  GaussHermite(){ }
  GaussHermite(string path, string basefn, size_t order);
  ~GaussHermite(){ }

  string GetPath()  const { return ghpath;       }
  string GetBase()  const { return ghbase;       }
  size_t GetOrder() const { return roots.size(); }
  const double GetRoot(size_t i) const{
    return (i<roots.size() ? roots[i] : 0.0); }
  const double GetWeight(size_t i) const{
    return (i<weights.size() ? weights[i] : 0.0); }
  const double GetRootSqrt2(size_t i) const{
    return (i<roots_sqrt2.size() ? roots_sqrt2[i] : 0.0); }
  
  const vector<double>& GetRoots()      const { return roots;   }
  const vector<double>& GetWeights()    const { return weights; }
  const vector<double>& GetRootsSqrt2() const{ return roots_sqrt2; }

protected:
  vector<double> ReadFile(string fname);
  string ghpath;
  string ghbase;
  vector<double> roots;
  vector<double> weights;
  vector<double> roots_sqrt2;
};


class GaussHermite2D : public GaussHermite{
public:
  GaussHermite2D(){ }
  GaussHermite2D(string path, string basefn, size_t order);
  ~GaussHermite2D(){ }

  const double GetWeightProd(size_t i, size_t j) const;
  const double GetRootDotProd(size_t i, size_t j,
			      const double& v0, const double& v1)const;

protected:
  vector<vector<double> > weightProd;
};


class GaussHermiteND : public GaussHermite{
public:
  GaussHermiteND(){ }
  GaussHermiteND(string path, string basefn, size_t order, size_t dim);
  ~GaussHermiteND(){ }

  const size_t GetDimension() const{ return powers.size()-1; }
  const double GetWeightProd(const vector<size_t>& i) const;
  const double GetWeightProd(const size_t& i) const;
  const vector<double>& GetWeightProd() const{ return weightProd; }
  const vector<size_t>& GetPowers() const{ return powers; }
  const void GetRootPermutation(vector<double>& v, const size_t& i) const;
  const void GetRootSqrt2Permutation(vector<double>& v, const size_t& i) const;
  const double GetRootDotProd(const vector<size_t>& i,
			      const vector<double>& v) const;

protected:
  vector<size_t> powers;
  vector<double> weightProd;
};

#endif
