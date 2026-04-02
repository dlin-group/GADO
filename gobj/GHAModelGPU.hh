#ifndef __GHAModelGPU__
#define __GHAModelGPU__

#include "GaussHermite.hh"
#include "PMatrix.hh"
#include "GMatrix.hh"
#include "BGMatrix.hh"
#include <vector>
#include <map>
#include <utility>
#include <cublas_v2.h>

using namespace std;

class GHAModelGPU{
public:
  
  GHAModelGPU(string ghpath, string ghbase, size_t ghdim, size_t ghorder,
	      GMatrix<double>* t, GMatrix<double>* d,
	      GMatrix<double>* x, BGMatrix<double>* p, cublasHandle_t chandle);
  virtual ~GHAModelGPU();

  virtual GaussHermiteND GetGaussHermite(size_t order) const;

  virtual size_t GetNIterations()   const { return niterations;       }
  virtual double GetEpsilon()       const { return epsilon;           }
  virtual void SetEpsilon(double e)       { epsilon = e;              }
  virtual bool GetBLast()           const { return b_last;            }
  virtual void SetBLast(bool bl)          { b_last = bl;              }
  virtual bool GetBFullNR()         const { return b_fullnr;          }
  virtual void SetBFullNR(bool nr)        { b_fullnr = nr;            }
  virtual bool GetVerbose()         const { return verbose;           }
  virtual void SetVerbose(bool v)         { verbose = v;              }
  
  virtual size_t GetNParam() const { return npar; }
  virtual size_t GetNIndiv() const { return nind; }
  
  virtual double           GetTau()        const { return tau;       }
  virtual double           GetUTau()       const { return Utau;      }
  virtual double           GetVTau()       const { return Vtau;      }
  virtual PMatrix<double>* GetBeta()       const { return beta;      }
  virtual PMatrix<double>* GetUBeta()      const { return Ubeta;     }
  virtual PMatrix<double>* GetVBeta()      const { return Vbeta;     }
  virtual GMatrix<double>* GetDelta()      const { return delta;     }
  virtual GMatrix<double>* GetTObs()       const { return T;         }
  virtual GMatrix<double>* GetXData()      const { return X;         }
  virtual PMatrix<double>* GetLambda()     const { return lambda_in; }
  virtual GMatrix<double>* GetCumLambda()  const { return clambda_in;}
  virtual BGMatrix<double>* GetGRM()       const { return psi;       }
  virtual BGMatrix<double>* GetGRMInv()    const { return psiinv;    }
  
  virtual void Solve(size_t maxiterap=0, size_t maxiterin=0);
  
protected:

  virtual void SetSeedsAP();
  virtual void TimeSortDataAP();
  virtual void UpdateAP_aq();
  virtual void UpdateAP_ebxw1q();
  virtual void UpdateAP_w1q();
  virtual void UpdateAP_beta();
  virtual void UpdateAP_lambda();
  virtual void UpdateAP_clambda();
  virtual void UpdateAP_tau();
  virtual bool IterateAP(size_t niter=1);
  virtual bool CheckConvergenceAP();

  /*virtual void SetSeedsIn();
  virtual void TimeSortDataIn();
  virtual void UpdateIn_b();
  virtual void UpdateIn_mus();
  virtual void UpdateIn_ebxw1();
  virtual void UpdateIn_w1();
  virtual void UpdateIn_beta();
  virtual void UpdateIn_lambda();
  virtual void UpdateIn_clambda();
  virtual bool IterateIn_b(size_t niter);
  virtual bool IterateIn(size_t niter=1);
  virtual bool CheckConvergenceIn();*/
  
  vector<GaussHermiteND> gauss_hermite;
  vector<double*> gh_roots_sqrt2;
  vector<size_t*> gh_powers;
  
  double epsilon;
  bool b_last;
  bool b_fullnr;
  size_t niterations;
  bool verbose;
  map<string, double> titer_us;
  cublasHandle_t handle;
  
  size_t npar;
  size_t nind;

  double nll;                  // complete neg log-likelihood
  double nllprev;              // nll from previous iteration
  double grad;                 // gradient of nll wrt tau
  double gradprev;             // gradient in the previous iteration
  double tau;                  // variance applied to grm
  double tauprev;              // tau from previous iteration
  double Utau;                 // score for tau
  double Vtau;                 // information for tau

  // data with original structure
  GMatrix<double>* T_in;       // T tilde,  min(T, C)
  GMatrix<double>* delta_in;   // survival time < censoring time
  GMatrix<double>* X_in;       // covariate data
  BGMatrix<double>* psi_in;    // genetic relationship matrix
  BGMatrix<double>* psiinv_in; // inverse of grm
  GMatrix<double>* ebx_in;     // ith element is exp(beta dot X_i)
  GMatrix<double>* ebxw1_in;   // ith element is ebx_i*w1q_i
  GMatrix<double>* w1_in;
  GMatrix<double>* lambda_in;  // hazard function (jump size)
  GMatrix<double>* lambdaprev_in;
  GMatrix<double>* clambda_in; // cumulative hazard function
  GMatrix<double>* b_in;       // n-variate zero-mean normal random
  GMatrix<double>* bprev_in;   // b from previous iteration
  GMatrix<double>* ub_in;      // Lambda(T_i) * exp(btx_i+b_i)
  GMatrix<double>* mu_in;      // means  in gaussian approx of b cond density
  GMatrix<double>* sigma_in;   // sigmas in gaussian approx of b cond density
  GMatrix<double>* buffvec_in; // buffer vector (nind) for computations
  BGMatrix<double>* buffmat_in;// buffer matrix (nind x nind) for computations

  // time sorted data with original structure
  GMatrix<double>* T_in_sort;
  GMatrix<double>* X_in_sort;
  GMatrix<double>* delta_in_sort;
  GMatrix<double>* ebxw1_in_sort;
  vector<size_t> T_in_perm;
  vector<size_t> T_in_perm_t;

  // data with all pairs structure
  GMatrix<double>* T;          // T tilde,  min(T, C)
  GMatrix<double>* delta;      // survival time < censoring time
  GMatrix<double>* X;          // covariate data
  BGMatrix<double>* psi;       // genetic relationship matrix
  BGMatrix<double>* psiinv;    // inverse of grm
  BGMatrix<double>* sqrtpsi;   // sqrt of psi
  PMatrix<double>* lambda;
  PMatrix<double>* lambdaprev;
  GMatrix<double>* clambda;    // cumulative hazard function

  // time sorted all pairs data
  PMatrix<double>* T_sort;
  PMatrix<double>* X_sort;
  PMatrix<double>* delta_sort;
  PMatrix<double>* ebxw1q_sort;
  vector<size_t> T_perm;
  vector<size_t> T_perm_t;

  // data subsets with constant block size
  vector<size_t> offset_cbs;
  vector<size_t> boffset_cbs;
  vector<BGMatrix<double>*> psiinv_cbs;
  vector<BGMatrix<double>*> sqrtpsi_cbs;
  vector<BGMatrix<double>*> Ebb_cbs;
  vector<GMatrix<double>*> delta_cbs;
  vector<GMatrix<double>*> aq_cbs;
  vector<GMatrix<double>*> buff0_cbs;
  vector<GMatrix<double>*> buff1_cbs;
  vector<double*> dtpb_cbs;
  vector<double*> atpb_cbs;
  vector<double*> sf0_cbs;

  GMatrix<double>* aq;         // element-wise Lambda*e^{beta*X}
  GMatrix<double>* w1q;        // E(exp(bi)) approximated by gauss-hermite
  PMatrix<double>* beta;       // covariate parameters
  PMatrix<double>* betaprev;   // beta from previous iteration
  GMatrix<double>* ebx;        // ith element is exp(beta dot X_i)
  GMatrix<double>* ebxw1q;     // ith element is ebx_i*w1q_i
  PMatrix<double>* Ubeta;      // score function for beta
  PMatrix<double>* Vbeta;      // information matrix for beta
  
  vector<size_t> index_map;

};

#endif
