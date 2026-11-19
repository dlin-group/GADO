#ifndef __GHAModel__
#define __GHAModel__

#include "config_defs.hh"
#include "GaussHermite.hh"
#include "PMatrix.hh"
#include "BDMatrix.hh"
#include <vector>
#include <map>
#include <utility>
#include <limits>
#ifdef __CUDA__
#include "GMatrix.hh"
#include <cublas_v2.h>
#include <vector>
#endif

using namespace std;

class GHAModel{
public:
  // EXPERIMENT ONLY. Values are recalculated per variant and never feed
  // the association score, p-value, or production variance ratio.
  struct VRAuditRecord {
    bool valid = false;
    double raw = 0.0;
    double cross = 0.0;       // sum u_b*c_b / scale_b
    double correction = 0.0;  // sum c_b*c_b / scale_b
    double exact = 0.0;
    double working = 0.0;
    double block_proxy = 0.0; // heuristic only, NOT a variance estimator
    double no_tau_correction = 0.0; // diagnostic counterfactual only
    double ratio = 0.0;
    double ratio_block_proxy = 0.0;
    double algebra_error = 0.0;
    double orthogonality_max = 0.0;
    double dominant_block_fraction = 0.0;
    size_t dominant_block_size = 0;
    size_t dominant_block_events = 0;
    long long dominant_block_id = -1; // original GRM block ID (model order)
    double tau_adjust_coeff = 0.0;   // h_tau = t - c' H^-1 q
    double sum_asc2_scaled = 0.0;     // sum Asc_i^2 / scale_i
    double sum_v_asc_scaled = 0.0;    // sum v_i Asc_i / scale_i
    double tau_cross_term = 0.0;     // -2*h_tau*sum(v_i*Asc_i/scale_i)
    double tau_squared_term = 0.0;   // h_tau^2*sum(Asc_i^2/scale_i)
    double tau_decomposition_error = 0.0;
    // Analytic/fixed-nuisance numerical tau Jacobian audit (not used in inference).
    double I_gamma_tau_analytic = std::numeric_limits<double>::quiet_NaN();
    double I_gamma_tau_fd = std::numeric_limits<double>::quiet_NaN();
    double tau_projection_analytic = std::numeric_limits<double>::quiet_NaN();
    double tau_projection_fd_c = std::numeric_limits<double>::quiet_NaN();
    double h_tau_fd_c = std::numeric_limits<double>::quiet_NaN();
    double I_gamma_tau_gap = std::numeric_limits<double>::quiet_NaN();
    double h_tau_fd_c_gap = std::numeric_limits<double>::quiet_NaN();
    long long top_tau_block_id = -1;
    size_t top_tau_block_size = 0;
    size_t top_tau_block_events = 0;
    double top_tau_block_fraction = 0.0;
  };

  
  GHAModel(string ghpath, string ghbase, size_t ghdim,
	   size_t ghorder, bool ghstep,
	   PMatrix<double>* t, PMatrix<double>* d,
	   PMatrix<double>* x, BDMatrix<double>* p);
  virtual ~GHAModel();
  virtual bool CheckConvergenceAP_TwoStage(bool freeze_bl, double eps_bl);
  virtual GaussHermiteND GetGaussHermite(size_t dim) const;
  virtual size_t GetGHDim()           const { return gauss_hermite.size(); }
  virtual size_t GetGHStepSize()      const { return gh_step;  }
  virtual size_t GetGHStepIter(size_t i) const {
    return i<gh_step_iter.size() ? gh_step_iter[i] : 0; }
  virtual size_t GetGHStepMax()              const { return gh_step_max;      }
  virtual void   SetGHStepMax(size_t m) { gh_step_max = std::max(m, gh_step); }
  virtual size_t GetGHStepIterConv()         const { return gh_step_iter_conv;}
  virtual void   SetGHStepIterConv(size_t c)       { gh_step_iter_conv = c;   }
  
  virtual size_t GetNIterations()                const { return niterations;  }
  virtual size_t GetNIterationsAP()              const {return niterations_ap;}
  virtual size_t GetNIterationsIn()              const {return niterations_in;}
  virtual bool GetConvergedAP()                  const { return converged_ap; }
  virtual void SetConvergedAP(bool c)                  { converged_ap = c;    }
  virtual bool GetConvergedIn()                  const { return converged_in; }
  virtual void SetConvergedIn(bool c)                  { converged_in = c;    }
  virtual double GetEpsilon()                    const { return epsilon;      }
  virtual void SetEpsilon(double e)                    { epsilon = e;         }
  virtual bool GetAbsoluteThresh()               const { return absolute_thresh;}
  virtual void SetAbsoluteThresh(bool a)               { absolute_thresh = a; }
  virtual double GetTauLimit()                   const { return taulimit;     }
  virtual void SetTauLimit(double t)                   { taulimit = t;        }
  virtual bool GetBLast()                        const { return b_last;       }
  virtual void SetBLast(bool bl)                       { b_last = bl;         }
  virtual bool GetBFullNR()                      const { return b_fullnr;     }
  virtual void SetBFullNR(bool nr)                     { b_fullnr = nr;       }
  virtual bool GetCentralizeG()                  const { return centralize_G; }
  virtual void SetCentralizeG(bool c)                  { centralize_G = c;    }
  virtual bool GetUnitaryTransformG()            const { return utransform_G; }
  virtual void SetUnitaryTransformG(bool u)            { utransform_G = u;    }
  virtual bool GetBNormUgamma()                  const { return bnorm_ugamma; }
  virtual void SetBNormUgamma(bool b)                  { bnorm_ugamma = b;    }
  virtual bool GetCacheSthetaIgamma()            const { return cache_SI;     }
  virtual void SetCacheSthetaIgamma(bool c)            { cache_SI = c;        }
  virtual bool GetVerbose()                      const { return verbose;      }
  virtual void SetAPDiagnostics(bool enabled) { ap_audit = enabled; }
  virtual void SetAPTauSeed(double value) { ap_tau_seed = value; }
  // Skip Step 1 and fit Step 2 at a known tau. Experimental mode only.
  virtual void SetExperimentalFixedTau(double value) {
    fixed_tau_mode = true; fixed_tau_value = value;
  }
  virtual bool GetExperimentalFixedTau() const { return fixed_tau_mode; }
  virtual void SetVRAuditEnabled(bool enabled) { vr_audit_enabled = enabled; }
  // Diagnostic only; writes prefix_step1_summary.tsv and prefix_step1_blocks.tsv
  virtual void WriteTauIFAudit(const string& prefix) const;
  // Diagnostic only: numerical tau derivative of the IMPLEMENTED posterior
  // weights at fixed (beta, baseline hazard), recomputing conditional b modes.
  // Must run after loading or computing Step 2 score/information, BEFORE SNPs.
  virtual void SetTauFDStep(double step) { tau_fd_step = step; }
  // Fixed-nuisance finite-difference tau cross-information for the block-
  // sandwich variance; the analytic H and q matrices remain unchanged.
  // Keep the old virtual setter for ABI/source compatibility; new code uses
  // SetTauFDCross() without introducing a new virtual-table entry.
  virtual void SetExperimentalTauFDCross(bool enabled) { tau_fd_cross_enabled = enabled; }
  void SetTauFDCross(bool enabled) { SetExperimentalTauFDCross(enabled); }
  virtual void SetTauFDFastW1(bool enabled) { tau_fd_fast_w1 = enabled; }
  virtual void PrepareTauFD(const string& prefix);
  virtual void ComputeIn_EV_W1Only(); // FD-only; no higher posterior moments written
  virtual void SetHQFDAuditEnabled(bool v) { hq_fd_audit_enabled=v; }
  virtual void RunHQFDAudit(const string& prefix,double eps); // CPU-only; first SNP, fixed tau/genotype residuals
  virtual const VRAuditRecord& GetVRAuditRecord() const {
    return vr_audit_record;
  }
  virtual void SetVerbose(bool v)                      { verbose = v;         }

  virtual double GetBetaRidge()                  const { return beta_ridge;     }
  virtual double GetBetaRidgeMin()               const { return beta_ridge_min; }
  virtual void   SetBetaRidgeMin(double r)             { beta_ridge_min = r;    }
  virtual double GetBetaRidgeMax()               const { return beta_ridge_max; }
  virtual void   SetBetaRidgeMax(double r)             { beta_ridge_max = r;    }
  virtual double GetBetaWeightMin()              const { return beta_weight_min;}
  virtual void   SetBetaWeightMin(double w)            { beta_weight_min = w;   }
  virtual double GetBetaWeightMax()              const { return beta_weight_max;}
  virtual void   SetBetaWeightMax(double w)            { beta_weight_max = w;   }
  virtual double GetBetaStepMax()                const { return beta_step_max;  }
  virtual void   SetBetaStepMax(double s)              { beta_step_max = s;     }
  virtual double GetBetaStepL2Max()              const { return beta_step_l2;   }
  virtual void   SetBetaStepL2Max(double l)            { beta_step_l2 = l;      }
  virtual double GetBetaMax()                    const { return beta_max;       }
  virtual void   SetBetaMax(double b)                  { beta_max = b;          }
  virtual double GetBetaXMax()                   const { return betax_max;      }
  virtual void   SetBetaXMax(double b)                 { betax_max = b;         }
  
  virtual bool GetUseVarRatio()                  const { return use_var_ratio;}
  virtual void SetUseVarRatio(bool v)                  { use_var_ratio = v;   }
  virtual double GetSPAThreshold()               const { return spa_thresh;   }
  virtual void SetSPAThreshold(double t)               { spa_thresh = t;      }
  virtual size_t GetVarRatioInitEvals()          const { return vr_init_eval; }
  virtual void   SetVarRatioInitEvals(size_t e)        { vr_init_eval = e;    }
  virtual size_t GetVarRatioStepEvals()          const { return vr_step_eval; }
  virtual void   SetVarRatioStepEvals(size_t s)        { vr_step_eval = s;    }
  virtual double GetVarRatioEpsilon()            const { return vr_epsilon;   }
  virtual void   SetVarRatioEpsilon(double e)          { if(e > 0.0 && e < 1.0) vr_epsilon = e; }
  // Defaults for the global ratio: 30 initial markers, +10 per update, two
  // consecutive stable median changes, maximum 500 valid markers.
  // No additional minimum marker count is enforced.
  virtual size_t GetVarRatioMaxEvals()           const { return vr_max_eval; }
  virtual void   SetVarRatioMaxEvals(size_t n)         { vr_max_eval = n;    }
  virtual size_t GetVarRatioStableChecks()       const { return vr_stable_checks; }
  virtual void   SetVarRatioStableChecks(size_t n)     { vr_stable_checks = std::max<size_t>(n, 1); }
  virtual bool   GetVarRatioDiagnostics()        const { return vr_diagnostics; }
  virtual void   SetVarRatioDiagnostics(bool v)        { vr_diagnostics = v; }
  virtual size_t GetVarRatioStableCount(size_t b) const {
    return b < vr_stable_count.size() ? vr_stable_count[b] : 0; }


  // ============================================================
  // One global variance ratio over calibration MAF [0.001, 0.5].
  // The one-element vector containers are retained for compatibility with
  // existing GHAIO HDF5 writers; no MAF-specific calibration is performed.
  // Variants outside the calibration range can still be association-tested.
  // IMPORTANT: GHAModel only sees genotype vectors (SetG), not call/INFO/QC
  // metadata. The upstream calibration driver must select high-quality,
  // QC-passed, directly genotyped markers from the GRM marker source.
  // ============================================================
  virtual size_t GetNVarRatioBins()              const { return vr_mean.size();}
  virtual const vector<double>& GetVarRatioMAFEdges() const {
    return vr_maf_edges; }
  // NOTE: this resets every per-bin value, call it BEFORE SetVarRatio()
  virtual void   SetVarRatioMAFEdges(const vector<double>& edges);
  virtual size_t GetVarRatioBin(double maf)      const;
  virtual size_t GetCurrentVarRatioBin()         const { return vr_bin;       }

  virtual double GetVarRatio(size_t b) const {
    return b < vr_mean.size() ? vr_mean[b] : 0.0; }
  virtual double GetVarRatio()                   const { return GetVarRatio(vr_bin); }
  virtual void   SetVarRatio(size_t b, double vr){
    if(b >= vr_mean.size()) return;
    vr_mean[b] = vr; var_ratio_found[b] = true; var_ratio_user[b] = true; }
  // Backward-compatible scalar setter (one-element calibration vector).
  virtual void   SetVarRatio(double vr){
    for(size_t b=0; b<vr_mean.size(); b++) SetVarRatio(b, vr); }

  virtual const vector<double>& GetVarRatioEstimates(size_t b) const {
    return var_ratio[b]; }
  // every bin's samples concatenated, kept so that the existing HDF5
  // writers keep compiling.  the bin an entry came from is lost here,
  // so use the indexed overload for anything quantitative.
  virtual const vector<double>& GetVarRatioEstimates() const;
  virtual bool GetVarRatioConverged(size_t b) const {
    return b < var_ratio_found.size() && var_ratio_found[b]; }
  virtual bool GetVarRatioConverged()            const { return AllVarRatioFound(); }
  virtual bool AllVarRatioFound()                const;
  
  virtual size_t GetNParam()                      const { return npar;        }
  virtual size_t GetNIndiv()                      const { return nind;        }
  virtual double GetTau()                         const { return tau;         }
  virtual void   SetTau(double t)                       { tau = t;            }
  virtual const PMatrix<double>* GetBeta()        const { return beta;        }
  virtual vector<double> GetTauIter()             const { return tau_iter;    }
  virtual vector<vector<double> > GetBetaIter()   const { return beta_iter;   }

  virtual size_t GetFitAP_ns()                    const { return ns_fit_ap;   }
  virtual size_t GetFitIn_ns()                    const { return ns_fit_in;   }
  virtual size_t GetScoreInfoAP_ns()              const { return ns_si_ap;    }
  virtual void SetScoreInfoAP_ns(size_t n)              { ns_si_ap = n;       }
  virtual size_t GetScoreInfoIn_ns()              const { return ns_si_in;    }
  virtual void SetScoreInfoIn_ns(size_t n)              { ns_si_in = n;       }
  virtual size_t GetScoreTest_ns()                const { return ns_st;       }
  virtual void SetScoreTest_ns(size_t n)                { ns_st = n;          }
  
  // accsessors for data in the input structure
  virtual const PMatrix<double>* GetBetaIn()      const { return beta1;       }
  virtual const PMatrix<double>* GetTIn()         const { return T_in;        }
  virtual const PMatrix<double>* GetXIn()         const { return X_in;        }
  virtual const PMatrix<double>* GetGIn()         const { return G_in;        }
  virtual const PMatrix<double>* GetDeltaIn()     const { return delta_in;    }
  virtual const PMatrix<double>* GetLambdaIn()    const { return lambda_in;   }
  virtual const PMatrix<double>* GetCumLambdaIn() const { return clambda_in;  }
  virtual const PMatrix<double>* GetbIn()         const { return b_in;        }
  virtual const BDMatrix<double>* GetGRMIn()      const { return psi_in;      }
  virtual const BDMatrix<double>* GetGRMInInv()   const { return psiinv_in;   }

  virtual void SetBetaIn(PMatrix<double>* b1)
  { if(b1) beta1->SetData(b1->GetData()); }
  virtual void SetLambdaIn(PMatrix<double>* l)
  { if(l) lambda_in->SetData(l->GetData()); }
  virtual void SetbIn(PMatrix<double>* bin)
  { if(bin) b_in->SetData(bin->GetData()); }

  // accessors for time-sorted data in the input structure
  virtual const PMatrix<double>* GetTInSort()     const { return T_in_sort;   }
  virtual const PMatrix<double>* GetXInSort()     const { return X_in_sort;   }
  virtual const PMatrix<double>* GetDeltaInSort() const {return delta_in_sort;}
  virtual const vector<size_t>&  GetTInPerm()     const { return T_in_perm;   }
  virtual const vector<size_t>&  GetTInPermTrans()const { return T_in_perm_t; }

  // accessors for score and information for the input structure step
  virtual const PMatrix<double>* GetTunq()        const { return T_unq;       }
  virtual const PMatrix<double>* GetUbetaIn()     const { return Ubeta_in;    }
  virtual const PMatrix<double>* GethIn()         const { return h_unq_in;    }
  virtual const PMatrix<double>* GetUhIn()        const { return Uh_unq_in;   }
  virtual const PMatrix<double>* GetInfoIn()      const { return V_unq_in;    }
  virtual const PMatrix<double>* GetInfoInvIn()   const { return Vinv_unq_in; }
  virtual const double           GetUtauIn()      const { return utau_in;     }
  virtual const vector<size_t>&  GetTunqIndexIn() const { return index_unq_in;}
  virtual const PMatrix<double>* GetebxIn()       const { return ebx_in;      }
  virtual const PMatrix<double>* Getw1In()        const { return w1_in;       }
  virtual const PMatrix<double>* Getebxw1In()     const { return ebxw1_in;    }
  virtual const BDMatrix<double>* GetEbbIn()      const { return Ebb_in;      }
  virtual const BDMatrix<double>* GetEebbIn()     const { return Eebb_in;     }
  virtual const PMatrix<double>*  GetEbbebIn()    const { return Ebbeb_in;    }

  // accessors for data in the all-pairs structure
  virtual const PMatrix<double>* GetBetaAP()      const { return beta0;       }
  virtual const PMatrix<double>* GetTAP()         const { return T;           }
  virtual const PMatrix<double>* GetXAP()         const { return X;           }
  virtual const PMatrix<double>* GetGAP()         const { return G;           }
  virtual const PMatrix<double>* GetDeltaAP()     const { return delta;       }
  virtual const PMatrix<double>* GetLambdaAP()    const { return lambda;      }
  virtual const PMatrix<double>* GetCumLambdaAP() const { return clambda;     }
  virtual const BDMatrix<double>* GetGRMAP()      const { return psi;         }
  virtual const BDMatrix<double>* GetGRMAPInv()   const { return psiinv;      }
  virtual const BDMatrix<double>* GetGRMAPSqrt()  const { return sqrtpsi;     }

  virtual void SetBetaAP(PMatrix<double>* b0)
  { if(b0) beta0->SetData(b0->GetData()); }
  virtual void SetLambdaAP(PMatrix<double>* l)
  { if(l) lambda->SetData(l->GetData()); }

  // accessors for time-sorted data in the all-pairs structure
  virtual const PMatrix<double>* GetTAPSort()     const { return T_sort;      }
  virtual const PMatrix<double>* GetXAPSort()     const { return X_sort;      }
  virtual const PMatrix<double>* GetDeltaAPSort() const { return delta_sort;  }
  virtual const vector<size_t>&  GetTAPPerm()     const { return T_perm;      }
  virtual const vector<size_t>&  GetTAPPermTrans()const { return T_perm_t;    }
  virtual const vector<size_t>& GetIndexMap()     const { return index_map;   }

  // accessors for score and information for the all-pairs step
  virtual const PMatrix<double>* GetUbetaAP()     const { return Ubeta;       }
  virtual const PMatrix<double>* GethAP()         const { return h_unq;       }
  virtual const PMatrix<double>* GetUhAP()        const { return Uh_unq;      }
  virtual const PMatrix<double>* GetInfoAP()      const { return V_unq;       }
  virtual const PMatrix<double>* GetInfoInvAP()   const { return Vinv_unq;    }
  virtual const double           GetUtauAP()      const { return utau;        }
  virtual const PMatrix<double>* GetAsc()         const { return Asc;         }
  virtual const vector<size_t>&  GetTunqIndexAP() const { return index_unq;   }
  virtual const PMatrix<double>* GetebxAP()       const { return ebx;         }
  virtual const PMatrix<double>* Getw1AP()        const { return w1q;         }
  virtual const PMatrix<double>* Getebxw1AP()     const { return ebxw1q;      }
  virtual const vector<BDMatrix<double>*> GetEbbAP()  const{ return Ebb_cbs;  }
  virtual const vector<BDMatrix<double>*> GetEebbAP() const{ return Eebb_cbs; }
  virtual const vector<BDMatrix<double>*> GetEbbbbAP()const{ return Ebbbb_cbs;}
  virtual const vector<vector<BDMatrix<double>* > >
  GetEbbebAP() const{ return Ebbeb_cbs; }
  
  // ============================================================
  // Fast weighted genotype residualization. The exact genetic score and
  // exact block variance retain centered G; working VR variance and the SPA
  // CGF consistently use G_tilde.
  // G_tilde = Gc - Xaug (Xaug' W Xaug)^(-1) Xaug' W Gc
  // ============================================================
  PMatrix<double>* XtW_resid;
  PMatrix<double>* XtWX_inv_resid;
  PMatrix<double>* XtWG_resid;
  PMatrix<double>* coefG_resid;
  PMatrix<double>* G_tilde_in;
  bool residualizer_ready;
  bool g_projection_ready;
  bool g_tilde_ready;
  double g_working_var;  // cached G_tilde' W G_tilde (same value for VR and SPA)
  std::vector<double> g_resid_coef;
  
  void PrepareGResidualizer();
  double ComputeResidualizedWorkingVar();
  void BuildResidualizedG();
  

  
  
  virtual const double& GetPVal()          const { return pval;            }
  virtual const double& GetPValCent()      const { return pval_cent;       }
  virtual const double& GetScoreStat()     const { return score_stat;      }
  virtual const double& GetScoreStatCent() const { return score_stat_cent; }
  virtual const double& GetUgamma()        const { return ugamma;          }
  virtual const double& GetUgammaCent()    const { return ugamma_cent;     }
  virtual const double& GetUgammaVar()     const { return ugamma_var;      }
  

  virtual void SetSeedsAP();
  virtual void SetSeedsIn();
  virtual void UpdateAP_ebxw1q();
  virtual void UpdateIn_ebxw1();
  virtual void Solve(size_t maxiterap=0, size_t maxiterin=0);
  virtual void SetG(PMatrix<double>* g);
  virtual double GetMeanG() const { return G_mean; }
  virtual double GetMAF()   const { return MAF;    }
  virtual double GetMAC()   const { return MAC;    }
  virtual void SetGamma(double g){ gamma = g; }
  virtual void ComputeScoreInfo();
  virtual void ComputeScoreTest(bool print=true);

  #ifdef __CUDA__
  virtual void SetCublasHandle(cublasHandle_t h){ chandle = h; }
  virtual cublasHandle_t GetCublasHandle(){ return chandle;    }
  #endif

  // I/O methods
  void WriteNullModelHDF5(string filename);
  void WriteNullModelHDF5(H5::H5File* outfile);

  void ReadNullModelHDF5(string filename);
  void ReadNullModelHDF5(H5::H5File* infile);

  void WriteScoreInfoHDF5(string filename);
  void WriteScoreInfoHDF5(H5::H5File* outfile);

  void ReadScoreInfoHDF5(string filename);
  void ReadScoreInfoHDF5(H5::H5File* infile);

  
protected:

  // methods used by both the all-pairs and input data steps
  virtual void TimeSortData(string label,
			    vector<size_t>&  p, vector<size_t>&  p_t,
			    PMatrix<double>* t, PMatrix<double>* t_sort,
			    PMatrix<double>* x, PMatrix<double>* x_sort,
			    PMatrix<double>* d, PMatrix<double>* d_sort,
			    PMatrix<double>* e, PMatrix<double>* e_sort);
  virtual void UpdateBeta(string label,
			  PMatrix<double>* x,      PMatrix<double>* expbx,
			  PMatrix<double>* t_sort, PMatrix<double>* x_sort,
			  PMatrix<double>* d_sort, PMatrix<double>* e_sort);
  virtual void UpdateLambda(string label,
			    PMatrix<double>* l,      PMatrix<double>* lprev,
			    PMatrix<double>* t_sort, PMatrix<double>* d_sort,
			    PMatrix<double>* e_sort);
  virtual void UpdateCLambda(string label,            vector<size_t>& p_t,
			     PMatrix<double>* l,      PMatrix<double>* c,
			     PMatrix<double>* t_sort);
  virtual pair<PMatrix<double>*, PMatrix<double>*>
  UpdateUnique(string label,
	       vector<size_t>& imap,    PMatrix<double>* d_sort,
	       PMatrix<double>* t_sort, PMatrix<double>* t_unq,
	       PMatrix<double>* l,      PMatrix<double>* l_unq);

  // methods specific to the all-pairs data step
  virtual void UpdateAP_aq();
  virtual void UpdateAP_w1q();
  virtual void UpdateAP_tau();
  virtual bool IterateAP(size_t niter=1);
  virtual double GetParamDiff(double val, double prev) const;
  virtual double GetMatrixDiff(PMatrix<double>* val, PMatrix<double>* prev)const;
  virtual bool CheckConvergenceAP();
  virtual void ComputeAP_EV();
  virtual void ScoreAP_beta();
  virtual void ScoreAP_h();
  virtual void ScoreAP_tau();
  virtual void InformationAP();
  virtual void ScoreInfoAP();

  // methods specific to the input data step
  virtual void UpdateIn_b();
  virtual void UpdateIn_mus();
  virtual void UpdateIn_w1();
  virtual bool IterateIn_b(size_t niter);
  virtual bool IterateIn(size_t niter=1);
  virtual bool CheckConvergenceIn();
  virtual void ComputeIn_EV();
  virtual void ComputeIn_EV_GH(size_t q);
  virtual void ScoreIn_beta();
  virtual void ScoreIn_h();
  virtual void ScoreIn_tau();
  virtual void InformationIn();
  virtual void ScoreInfoIn();

  virtual void InitVarRatioBins();   // (re)allocate the per-bin state
  virtual double ComputeVarRatio();  // returns rhat[bin]*GUG
  virtual double ComputePValue(const double& ug, const double& var, const double& var_ratio); //p-value, saddle-point approx
  virtual double ComputeWeightedP(double ug, double var, double vr);
  virtual void ComputeCGFs(const double& c1,
			   double& k0, double& k1, double& k2);
  
  virtual void StridedInnerProduct(PMatrix<double>* v0, PMatrix<double>* v1,
				   size_t stride, vector<double>& out,
				   double off0=0.0, double off1=0.0,
				   bool snorm=false);
  virtual void StridedInnerProduct(PMatrix<double>* v0, PMatrix<double>* v1,
				   const vector<size_t>& stride,
				   vector<double>& out,
				   double off0=0.0, double off1=0.0,
				   bool snorm=false);

  vector<GaussHermiteND> gauss_hermite;
  size_t gh_step;
  size_t gh_step_max;
  size_t gh_step_iter_conv;
  vector<size_t> gh_step_iter;
  double epsilon;
  double taulimit;
  double beta_ridge;
  double beta_ridge_min;
  double beta_ridge_max;
  double beta_weight_min;
  double beta_weight_max;
  double beta_step_max;
  double beta_step_l2;
  double beta_max;
  double betax_max;
  bool absolute_thresh;
  bool b_last;
  bool b_fullnr;
  bool centralize_G;
  bool utransform_G;
  bool bnorm_ugamma;
  size_t niterations;
  size_t niterations_ap;
  size_t niterations_in;
  bool converged_ap;
  bool converged_in;
  bool verbose;
  map<string, double> titer_us;
  map<string, double> titer_ist_us;
  size_t ns_fit_ap;
  size_t ns_fit_in;
  size_t ns_si_ap;
  size_t ns_si_in;
  size_t ns_st;
  
  size_t npar;
  size_t nind;

  vector<double> tau_iter;           // values of tau for each iteration
  vector<vector<double> > beta_iter; // values of beta for each iteration

  double tau;                  // variance applied to grm
  double tauprev;              // tau from previous iteration
  double gamma;                // genetic variable parameter
  PMatrix<double>* beta;       // covariate parameters
  PMatrix<double>* betaprev;   // beta from previous iteration
  PMatrix<double>* beta0;      // beta result from the all-pairs step
  PMatrix<double>* beta1;      // beta result from the input data step
  PMatrix<double>* Ubeta;      // score function for beta
  PMatrix<double>* Vbeta;      // information matrix for beta

  // data with original structure
  PMatrix<double>* T_in;       // T tilde,  min(T, C)
  PMatrix<double>* delta_in;   // survival time < censoring time
  PMatrix<double>* X_in;       // covariate data
  PMatrix<double>* G_in;       // genetic variable
  BDMatrix<double>* psi_in;    // genetic relationship matrix
  BDMatrix<double>* psiinv_in; // inverse of grm
  PMatrix<double>* ebx_in;     // ith element is exp(beta dot X_i)
  PMatrix<double>* xb_in;     // ith element is beta dot X_i
  PMatrix<double>* ebxw1_in;   // ith element is ebx_i*w1q_i
  PMatrix<double>* w1_in;      // expectation of e^b
  PMatrix<double>* lambda_in;  // hazard function (jump size)
  PMatrix<double>* lambdaprev_in; // lambda from previous iteration
  PMatrix<double>* clambda_in; // cumulative hazard function
  PMatrix<double>* b_in;       // n-variate zero-mean normal random
  PMatrix<double>* bprev_in;   // b from previous iteration
  PMatrix<double>* ub_in;      // Lambda(T_i) * exp(btx_i+b_i)
  PMatrix<double>* mu_in;      // means  in gaussian approx of b cond density
  PMatrix<double>* sigma_in;   // sigmas in gaussian approx of b cond density
  PMatrix<double>* buffvec_in; // buffer vector (nind) for computations
  BDMatrix<double>* buffmat_in;// buffer matrix (nind x nind) for computations

  // time sorted data with original structure
  PMatrix<double>* T_in_sort;
  PMatrix<double>* X_in_sort;
  PMatrix<double>* delta_in_sort;
  PMatrix<double>* ebxw1_in_sort;
  vector<size_t> T_in_perm;
  vector<size_t> T_in_perm_t;

  
  // i'th element in all-pairs data is index_map[i] in input data
  vector<size_t> index_map;

  // data with all pairs structure
  PMatrix<double>* T;          // T tilde,  min(T, C)
  PMatrix<double>* X;          // covariate data
  PMatrix<double>* delta;      // survival time < censoring time
  PMatrix<double>* G;          // genetic variable
  vector<size_t> LM;           // link matrix
  BDMatrix<double>* psi;       // genetic relationship matrix
  BDMatrix<double>* psiinv;    // inverse of grm
  BDMatrix<double>* sqrtpsi;   // sqrt of psi
  PMatrix<double>* lambda;     // hazard function (jump size)
  PMatrix<double>* lambdaprev; // lambda from previous iteration
  PMatrix<double>* clambda;    // cumulative hazard function
  PMatrix<double>* aq;         // element-wise Lambda*e^{beta*X}
  PMatrix<double>* w1q;        // E(exp(bi)) approximated by gauss-hermite
  PMatrix<double>* ebx;        // ith element is exp(beta dot X_i)
  PMatrix<double>* ebxw1q;     // ith element is ebx_i*w1q_i

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
  vector<BDMatrix<double>*> psiinv_cbs;
  vector<BDMatrix<double>*> sqrtpsi_cbs;
  vector<PMatrix<double>*> delta_cbs;
  vector<PMatrix<double>*> aq_cbs;
  vector<PMatrix<double>*> buff0_cbs;
  vector<vector<PMatrix<double>*> > buff1_cbs;
  #ifdef __CUDA__
  double* w1q_dev;
  vector<double*> psiinv_cbs_dev;
  vector<double*> Ebb_cbs_dev;
  vector<double*> delta_cbs_dev;
  vector<double*> aq_cbs_dev;
  vector<double*> tau_cbs_dev;
  vector<double*> reduce_cbs_dev;
  vector<vector<double> > reduce_cbs;
  vector<double*> buff0_cbs_dev;
  vector<vector<double*> > buff1_cbs_dev;
  vector<vector<double*> > buff2_cbs_dev;
  vector<double*> dtpb_dev;
  vector<double*> atpb_dev;
  vector<double*> sf0_dev;
  #endif

  // expectation values used in the score and information calculations for AP
  vector<BDMatrix<double>*> Ebb_cbs;             // E(b*b)
  vector<BDMatrix<double>*> Eebb_cbs;            // E(e^{b+b})
  vector<BDMatrix<double>*> Ebbbb_cbs;           // E(b*b*Psi^{-1}*b*b)
  vector<vector<BDMatrix<double>* > > Ebbeb_cbs; // E(e^{b}*b*b)
  #ifdef __CUDA__
  double* ebx_dev;
  double* clambda_dev;
  double* X_dev;
  double* T_dev;
  double* T_unq_dev;
  vector<double*> Eebb_cbs_dev;
  vector<double*> S2_cbs_dev;
  double* V_unq_dev;
  PMatrix<double>* V_unq_buff;
  #endif
  
  // score and information for all-pairs step
  vector<size_t> index_unq;
  PMatrix<double>* T_unq;
  PMatrix<double>* h_unq;
  PMatrix<double>* Uh_unq;
  PMatrix<double>* V_unq;
  PMatrix<double>* Vinv_unq;
  double utau;

  // expectation values used in the score and information calculations for in
  BDMatrix<double>* Ebb_in;    // E(b*b)
  BDMatrix<double>* Eebb_in;   // E(e^{b+b})
  PMatrix<double>*  Ebbeb_in;  // E(e^{b}*b*Psi^{-1}*b)
  
  // score and information for input data step
  vector<size_t> index_unq_in;
  PMatrix<double>* Ubeta_in;
  PMatrix<double>* h_unq_in;
  PMatrix<double>* Uh_unq_in;
  PMatrix<double>* V_unq_in;
  PMatrix<double>* Vinv_unq_in;
  double utau_in;

  // score test
  double pval;
  double pval_cent;
  double score_stat;
  double score_stat_cent;
  double ugamma;
  double ugamma_cent;
  double ugamma_var;
  double MAF;
  double MAC;
  double G_mean;
  PMatrix<double>* Asc;
  bool cache_SI;
  vector<PMatrix<double>* > Stheta_cache;
  vector<PMatrix<double>* > Igamma_cache;
  PMatrix<double>* G_transform;
  #ifdef __CUDA__
  cublasHandle_t chandle;
  GMatrix<double>* gG;
  GMatrix<double>* gutG;
  GMatrix<double>* gStheta;
  GMatrix<double>* gIgamma;
  GMatrix<double>* gI_G;
  GMatrix<double>* gSthetaI_G;
  #endif

  // ============================================================
  // variance ratio, maintained as a one-element vector for HDF5 compatibility
  // ============================================================
  bool use_var_ratio;
  size_t vr_init_eval;
  size_t vr_step_eval;
  size_t vr_max_eval;
  size_t vr_stable_checks;
  double vr_epsilon;
  bool   vr_diagnostics;
  bool   ap_audit;
  double ap_tau_seed;
  bool   fixed_tau_mode;
  double fixed_tau_value;
  bool   vr_audit_enabled;
  bool   ap_scores_fresh;
  double tau_fd_step;
  bool tau_fd_ready;
  bool tau_fd_cross_enabled;
  bool tau_fd_fast_w1;
  bool hq_fd_audit_enabled;
  vector<double> hq_analytic_q;
  vector<double> tau_fd_dw1;
  vector<double> tau_fd_c; // fixed-nuisance numerical -dU_xi/dtau
  vector<double> tau_fd_direct_coef; // lambda*exp(X beta)*dw1/dtau
  VRAuditRecord vr_audit_record;
  double spa_thresh;

  vector<double> vr_maf_edges;       // bin boundaries, ascending
  size_t vr_bin;                     // bin of the variant currently in G_in
  vector<vector<double> > var_ratio; // per-bin exact-variance ratio samples
  vector<bool>   var_ratio_found;    // per-bin convergence flag
  vector<bool>   var_ratio_user;     // per-bin user-supplied flag
  vector<double> vr_mean;            // per-bin rhat
  vector<double> vr_prev;            // per-bin previous median
  vector<size_t> vr_stable_count;     // consecutive stable updates per bin
  bool vr_weights_ready;             // ub_in holds clambda_in*ebxw1_in
  mutable vector<double> var_ratio_all; // scratch for the no-arg accessor
};

#endif
