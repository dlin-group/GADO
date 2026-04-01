#ifndef __GRMModel__
#define __GRMModel__

#include "GaussHermite.hh"
#include "PMatrix.hh"
#include <vector>
#include <map>
#include <utility>
#include <cublas_v2.h>

using namespace std;

class GRMModel{
public:
  
  GRMModel(PMatrix<double>* t, PMatrix<double>* d,
	   PMatrix<double>* x, PMatrix<double>* p, PMatrix<double>* pinv=NULL);
  virtual ~GRMModel();

  virtual GaussHermite2D GetGaussHermite() const { return gauss_hermite; }
  virtual void LoadGaussHermite(string path, string basefn, size_t order);

  virtual size_t GetNIterations()   const { return niterations;       }
  virtual double GetEpsilon()       const { return epsilon;           }
  virtual void SetEpsilon(double e)       { epsilon = e;              }
  virtual bool GetVerbose()         const { return verbose;           }
  virtual void SetVerbose(bool v)         { verbose = v;              }
  virtual bool GetbPartialLaplace() const { return b_partial_laplace; }
  virtual void SetbPartialLaplace(bool pl){ b_partial_laplace = pl;   }
  virtual bool GetbFromLast()       const { return b_from_last;       }
  virtual void SetbFromLast(bool fl)      { b_from_last = fl;         }
  
  virtual size_t GetNParam() const { return npar; }
  virtual size_t GetNIndiv() const { return nind; }
  
  virtual double           GetTau()        const { return tau;       }
  virtual double           GetUTau()       const { return Utau;      }
  virtual double           GetVTau()       const { return Vtau;      }
  virtual PMatrix<double>* GetBeta()       const { return beta;      }
  virtual PMatrix<double>* GetUBeta()      const { return Ubeta;     }
  virtual PMatrix<double>* GetVBeta()      const { return Vbeta;     }
  virtual PMatrix<double>* GetDelta()      const { return delta;     }
  virtual PMatrix<double>* GetTObs()       const { return T;         }
  virtual PMatrix<double>* GetXData()      const { return X;         }
  virtual PMatrix<double>* GetBRand()      const { return b;         }
  virtual PMatrix<double>* GetLambda()     const { return lambda;    }
  virtual PMatrix<double>* GetCumLambda()  const { return clambda;   }
  virtual PMatrix<double>* GetGRM()        const { return psi;       }
  virtual PMatrix<double>* GetGRMInv()     const { return psiinv;    }
  virtual double           GetGRMLogDet()  const { return psilogdet; }
  virtual double           GetNLL()        const { return nll;       }
  virtual vector<double>   GetTauNLL()     const { return taunll;    }
  virtual vector<vector<double> > GetBetaNLL() const { return betanll; }
  
  virtual void Solve(size_t maxiter=0, size_t nbiter=0);
  virtual void SolveGH2(size_t maxiter, size_t nbiter,
			vector<size_t>& bsize,
			vector<pair<size_t, size_t> >& sample);
  virtual void GridSearchTau(size_t maxiter, size_t maxbiter,
			     double tmin, double tmax, size_t ntau);
  virtual void GradientDescentTau(size_t maxgiter,
				  size_t maxiter, size_t maxbiter,
				  double tauseed, double stepseed);
  virtual void GridSearchTauGrad(size_t maxiter, size_t maxbiter,
				 double tmin, double tmax, size_t ntau);
  virtual double EvalNLL(double utau, vector<double>& ubeta);
  
protected:

  virtual void TimeSortData();
  virtual void SetSeeds();
  virtual void UpdateNLL();
  virtual void UpdateTauGradient();
  virtual void UpdateTauGradient2();
  virtual void Updateb();
  virtual void UpdateBeta();
  virtual void UpdateLambda();
  virtual void UpdateTau();
  virtual void UpdateTauGH2();
  virtual void UpdateTauGH2(vector<size_t>& bsize,
			    vector<pair<size_t, size_t> >& sample);
  virtual bool Iterateb(size_t niter=1);
  virtual bool Iterate(size_t niter=1, size_t nbiter = 1, bool updateTau=true);
  virtual bool IterateGH2(size_t niter, size_t nbiter,
			  vector<size_t>& bsize,
			  vector<pair<size_t, size_t> >& sample);
  virtual bool CheckConvergence();
  
  GaussHermite2D gauss_hermite;
  double epsilon;
  size_t niterations;
  bool verbose;
  map<string, double> titer_us;
  map<string, double> ttotal_us;
  
  size_t npar;
  size_t nind;

  bool b_partial_laplace;
  bool b_from_last;

  double nll;                  // complete neg log-likelihood
  double nllprev;              // nll from previous iteration
  double grad;                 // gradient of nll wrt tau
  double gradprev;             // gradient in the previous iteration
  double tau;                  // variance applied to grm
  double tauprev;              // tau from previous iteration
  double Utau;                 // score for tau
  double Vtau;                 // information for tau
  double psilogdet;            // log(|psi|)
  double tpiUlogdet;           // log(|(tau*psi)^-1 + U(b)|)
  double btpib;                // (b^T * psi^-1 * b) / (2*tau)
  PMatrix<double>* beta;       // covariate parameters
  PMatrix<double>* betaprev;   // beta from previous iteration
  PMatrix<double>* Ubeta;      // score function for beta
  PMatrix<double>* Vbeta;      // information matrix for beta
  PMatrix<double>* delta;      // survival time < censoring time
  PMatrix<double>* T;          // T tilde,  min(T, C)
  PMatrix<double>* X;          // covariate data
  PMatrix<double>* ebx;        // ith element is exp(beta dot X_i)
  PMatrix<double>* b;          // n-variate zero-mean normal random
  PMatrix<double>* bprev;      // b from previous iteration
  PMatrix<double>* ub;         // Lambda(T_i) * exp(btx_i+b_i)
  PMatrix<double>* lambda;     // hazard function (jump size)
  PMatrix<double>* lambdaprev; // lambda from previous iteration
  PMatrix<double>* clambda;    // cumulative hazard function
  PMatrix<double>* psi;        // genetic relationship matrix
  PMatrix<double>* psiinv;     // inverse of grm
  PMatrix<double>* w;          // E(exp(bi)) approximated by gauss-hermite
  PMatrix<double>* mu;         // means  in gaussian approx of b cond density
  PMatrix<double>* sigma2;     // sigmas in gaussian approx of b cond density
  PMatrix<double>* buffmat;    // buffer matrix (nind x nind) for computations
  PMatrix<double>* tpiU;       // (tau*psi)^-1 + U(b)
  PMatrix<double>* psiinvun;
  
  double taumin;
  double taumax;
  vector<double> taunll;
  vector<vector<double> > betanll;

  vector<size_t> permutation;
  vector<size_t> permutationt;
  
  bool delpsiinv;

  cublasHandle_t handle;
  int* pivot_dev;
  int* info_dev;
  double* b_dev;
  double* ub_dev;
  double* mu_dev;
  double* w_dev;
  double* delta_dev;
  double* psiinv_dev;
  double* tmp_dev;
  double* tpiU_dev;

  double** b_pntrs_dev;
  double** mu_pntrs_dev;
  double** w_pntrs_dev;
  double** psiinv_pntrs_dev;
  double** tmp_pntrs_dev;
  double** tpiU_pntrs_dev;
  
  double* tau_host;
  double* alpha_host;
  double* gamma_host;
  vector<double> tpiU_host;
  //virtual void tpiU_inverse();
};

#endif
