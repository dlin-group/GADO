#include "PMatrix.hh"
#include "BDMatrix.hh"
#include "GHAModel.hh"
#include "GHAIO.hh"
#include "SparseDosageReader.hh"
#include <H5Cpp.h>
#include <getopt.h>
#include <assert.h>
#include <cmath>
#include <iomanip>
#include <utility>
#include <chrono>
#include <cblas.h>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <numeric>
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <sstream>
#ifdef __CUDA__
#include <cublas_v2.h>
#endif

using namespace std;
using namespace std::chrono;

// Format console diagnostics without changing floating-point calculations or
// stream state. Four digits after the decimal; very small values use scientific
// notation so convergence residuals and GWAS p-values are not rounded to zero.
static std::string Log4(double value){
  std::ostringstream out;
  if(std::isfinite(value) && value != 0.0 && std::fabs(value) < 1.e-4)
    out << std::scientific;
  else
    out << std::fixed;
  out << std::setprecision(4) << value;
  return out.str();
}

// Only the GADO driver is available here (the SparseDosageReader class
// source was not uploaded). Its temporary metadata reader may print a
// misleading "read 0 ... 0 ns/variant" message on destruction. Capture that
// one-time progress message while preserving any other reader diagnostics.
class ScopedReaderLogCapture {
public:
  ScopedReaderLogCapture() : original_(std::cout.rdbuf(captured_.rdbuf())) {}
  ~ScopedReaderLogCapture() { std::cout.rdbuf(original_); }
  std::string str() const { return captured_.str(); }
  ScopedReaderLogCapture(const ScopedReaderLogCapture&) = delete;
  ScopedReaderLogCapture& operator=(const ScopedReaderLogCapture&) = delete;
private:
  std::ostringstream captured_;
  std::streambuf* original_;
};

static void ReplayNonProgressReaderMessages(const std::string& captured){
  std::istringstream input(captured);
  std::string line;
  while(std::getline(input,line)){
    // GADO reports its own properly labeled initialization and per-variant
    // reading progress. Discard only duplicated reader progress lines.
    if(line.compare(0,25,"SparseDosageReader: read ")==0) continue;
    std::cout << line << '\n';
  }
}

double median(std::vector<double>& v) {
  size_t n = v.size();
  size_t mid = n / 2;
  
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  double med = v[mid];
  
  if (n % 2 == 0) {
    std::nth_element(v.begin(), v.begin() + mid - 1, v.end());
    med = 0.5 * (med + v[mid - 1]);
  }
  
  return med;
}


// ------------------------------------------------------------------
// minimal HDF5 helpers for the per-MAF-bin variance ratios.
// kept local so that GHAIO does not have to change.
// ------------------------------------------------------------------
static bool ReadH5DoubleVector(H5::H5File& f, const string& name,
			       vector<double>& out){
  if(H5Lexists(f.getId(), name.c_str(), H5P_DEFAULT) <= 0) return false;
  H5::DataSet   ds = f.openDataSet(name.c_str());
  H5::DataSpace sp = ds.getSpace();
  hsize_t n = 1;
  if(sp.getSimpleExtentNdims() > 0) sp.getSimpleExtentDims(&n);
  if(n == 0) return false;
  out.assign(n, 0.0);
  ds.read(&out.front(), H5::PredType::NATIVE_DOUBLE);
  return true;
}

static void WriteH5DoubleVector(H5::H5File& f, const string& name,
				const vector<double>& v){
  if(v.empty()) return;
  if(H5Lexists(f.getId(), name.c_str(), H5P_DEFAULT) > 0)
    f.unlink(name.c_str());
  hsize_t n = v.size();
  H5::DataSpace sp(1, &n);
  H5::DataSet   ds = f.createDataSet(name.c_str(),
				     H5::PredType::NATIVE_DOUBLE, sp);
  ds.write(&v.front(), H5::PredType::NATIVE_DOUBLE);
}


// One-time, MAF-stratified random-order calibration pool builder.
// Pass 1: uniformly sample per MAF bin by reservoir sampling.
// Pass 2: stream original sparse HDF5 again, writing selected SNPs in a
//         globally shuffled output order (NOT grouped by MAF or chromosome).
// Memory: one genotype vector and one float output row, not 6000 vectors.
static void BuildRandomVRPool(const string& source, const string& output,
                              size_t per_bin, uint64_t seed,
                              const vector<double>& edges) {
  if(per_bin == 0) throw runtime_error("--vr-pool-size must be positive");
  if(edges.size() < 2) throw runtime_error("VR pool requires at least two MAF edges");
  const size_t nbins = edges.size() - 1;
  const size_t requested = nbins * per_bin;
  mt19937_64 rng(seed);
  vector<vector<size_t> > reservoirs(nbins);
  vector<size_t> eligible(nbins, 0);
  size_t nv=0, ni=0;
  vector<uint64_t> sample_ids;

  {
    SparseDosageReader reader(source);
    nv=reader.GetNVariants();
    ni=reader.GetNInd();
    sample_ids.assign(reader.GetIDs().begin(),reader.GetIDs().end());
    PMatrix<double> geno("vr_maf_scan",ni,1,false);
    for(size_t v=0; v<nv; ++v){
      if(!reader.NextVariant(&geno))
        throw runtime_error("Unexpected EOF during MAF scan");
      double sum=0.0;
      bool valid=true;
      for(size_t i=0;i<ni;++i){
        const double g=geno.GetVal(i);
        if(!std::isfinite(g) || g<0.0 || g>2.0){ valid=false; break; }
        sum+=g;
      }
      if(!valid) continue;
      const double af=sum/(2.0*static_cast<double>(ni));
      const double maf=std::min(af,1.0-af);
      if(!std::isfinite(maf) || maf<edges.front() || maf>edges.back()) continue;
      size_t bin = nbins - 1;
      for(size_t b=0; b<nbins; ++b){
        if(maf <= edges[b+1]){ bin=b; break; }
      }
      const size_t observed=++eligible[bin];
      auto& pool=reservoirs[bin];
      if(pool.size()<per_bin) pool.push_back(v);
      else{
        std::uniform_int_distribution<size_t> pick(0,observed-1);
        const size_t j=pick(rng);
        if(j<per_bin) pool[j]=v;
      }
      if((v+1)%20000==0)
        cout << "VR pool first pass: " << (v+1) << "/" << nv << " variants" << endl;
    }
  }
  for(size_t b=0;b<nbins;++b){
    cout << "Global VR pool: eligible=" << eligible[b]
         << ", sampled=" << reservoirs[b].size() << endl;
    if(reservoirs[b].size()!=per_bin)
      throw runtime_error("Insufficient QC-passed markers in MAF bin " + to_string(b)
          + ": found " + to_string(eligible[b]) + ", requested " + to_string(per_bin));
  }

  struct Choice{size_t src; uint32_t bin;};
  vector<Choice> choices;
  choices.reserve(requested);
  for(size_t b=0;b<nbins;++b)
    for(size_t idx:reservoirs[b]) choices.push_back({idx,static_cast<uint32_t>(b)});
  // Independent, global output permutation: avoids three consecutive MAF groups.
  shuffle(choices.begin(),choices.end(),rng);
  vector<pair<size_t,size_t> > lookup;
  lookup.reserve(requested);
  for(size_t j=0;j<requested;++j) lookup.emplace_back(choices[j].src,j);
  sort(lookup.begin(),lookup.end());

  H5File out(output.c_str(),H5F_ACC_TRUNC);
  hsize_t dims[2]={static_cast<hsize_t>(requested),static_cast<hsize_t>(ni)};
  DataSpace space(2,dims);
  DSetCreatPropList props;
  hsize_t chunk[2]={1,static_cast<hsize_t>(ni)};
  props.setChunk(2,chunk);
  props.setDeflate(1);
  DataSet ds=out.createDataSet("dosage",PredType::NATIVE_FLOAT,space,props);
  hsize_t nidx=requested;
  DataSpace idxspace(1,&nidx);
  DataSet idxds=out.createDataSet("source_variant_index",PredType::NATIVE_UINT64,idxspace);
  DataSet binds=out.createDataSet("maf_bin",PredType::NATIVE_UINT32,idxspace);
  vector<uint64_t> srcidx(requested);
  vector<uint32_t> binidx(requested);
  for(size_t j=0;j<requested;++j){srcidx[j]=choices[j].src;binidx[j]=choices[j].bin;}
  idxds.write(srcidx.data(),PredType::NATIVE_UINT64);
  binds.write(binidx.data(),PredType::NATIVE_UINT32);
  hsize_t nid=ni;
  DataSpace idspace(1,&nid);
  DataSet idsds=out.createDataSet("sample_ids",PredType::NATIVE_UINT64,idspace);
  idsds.write(sample_ids.data(),PredType::NATIVE_UINT64);
  hsize_t nedges=edges.size();
  DataSpace espace(1,&nedges);
  DataSet eds=out.createDataSet("maf_edges",PredType::NATIVE_DOUBLE,espace);
  eds.write(edges.data(),PredType::NATIVE_DOUBLE);

  SparseDosageReader reader(source);
  if(reader.GetNVariants()!=nv || reader.GetNInd()!=ni ||
     !equal(reader.GetIDs().begin(),reader.GetIDs().end(),sample_ids.begin()))
    throw runtime_error("Source genotype file changed between sampling and writing");
  PMatrix<double> geno("vr_write_geno",ni,1,false);
  vector<float> row(ni);
  size_t target=0;
  for(size_t v=0;v<nv && target<lookup.size();++v){
    if(!reader.NextVariant(&geno))
      throw runtime_error("Unexpected EOF during selected marker output");
    if(v!=lookup[target].first) continue;
    for(size_t i=0;i<ni;++i) row[i]=static_cast<float>(geno.GetVal(i));
    hsize_t offset[2]={static_cast<hsize_t>(lookup[target].second),0};
    hsize_t count[2]={1,static_cast<hsize_t>(ni)};
    DataSpace fspace=ds.getSpace();
    fspace.selectHyperslab(H5S_SELECT_SET,count,offset);
    DataSpace mspace(2,count);
    ds.write(row.data(),PredType::NATIVE_FLOAT,mspace,fspace);
    ++target;
    if(target%500==0)
      cout << "VR pool: written " << target << "/" << requested << endl;
  }
  if(target!=requested) throw runtime_error("Could not decode all selected calibration variants");
  cout << "Wrote MAF-stratified, globally shuffled VR pool " << output
       << ", markers per bin=" << per_bin
       << ", total=" << requested << ", individuals=" << ni
       << ", seed=" << seed << endl;
}

// Reads pre-shuffled calibration rows; genome-wide score testing can still use
// the original (unmodified) sparse HDF5 input.
static void CalibrateFromVRPool(const string& filename, GHAModel* model,
                                const vector<size_t>& permutation,
                                size_t n_model) {
  H5File input(filename.c_str(), H5F_ACC_RDONLY);
  DataSet ds=input.openDataSet("dosage");
  DataSpace sp=ds.getSpace();
  hsize_t dims[2]={0,0};
  if(sp.getSimpleExtentNdims()!=2) throw runtime_error("VR pool dosage must be a 2D matrix");
  sp.getSimpleExtentDims(dims);
  const size_t nvar=static_cast<size_t>(dims[0]);
  const size_t nsamp=static_cast<size_t>(dims[1]);
  if(n_model > nsamp || permutation.size() > nsamp)
    throw runtime_error("VR pool sample count is smaller than null-model sample count");
  // Verify sample identifiers after the same permutation used for GWAS.
  DataSet idds=input.openDataSet("sample_ids");
  DataSpace idsp=idds.getSpace();
  hsize_t idn=0; idsp.getSimpleExtentDims(&idn);
  if(idn != dims[1]) throw runtime_error("VR pool sample_ids length mismatch");
  // Sample ID mapping must be externally consistent across genotype and phenotype;
  // this preserves the exact existing GADO sample-permutation convention.
  vector<float> row(nsamp);
  PMatrix<double>* G = new PMatrix<double>("vr_pool_genotype",n_model,1,false);
  cout << "Calibrating global VR using shuffled pool..." << endl;
  for(size_t v=0; v<nvar && !model->AllVarRatioFound(); ++v){
    hsize_t offset[2]={static_cast<hsize_t>(v),0};
    hsize_t count[2]={1,static_cast<hsize_t>(nsamp)};
    DataSpace filespace=ds.getSpace();
    filespace.selectHyperslab(H5S_SELECT_SET,count,offset);
    DataSpace memspace(2,count);
    ds.read(row.data(),PredType::NATIVE_FLOAT,memspace,filespace);
    for(size_t j=0; j<n_model; ++j){
      const size_t original=(permutation.empty()?j:permutation.at(j));
      if(original >= nsamp) throw runtime_error("Invalid sample permutation for VR pool");
      G->SetVal(j,static_cast<double>(row[original]));
    }
    model->SetG(G);
    model->ComputeScoreTest(false);
  }
  if(!model->AllVarRatioFound())
    throw runtime_error("VR calibration pool exhausted before the global variance ratio converged; increase --vr-pool-size or inspect calibration markers");
  cout << "Shuffled VR calibration complete: rhat="
       << Log4(model->GetVarRatio(0)) << endl;
  // G belongs to model after SetG; GHAModel destructor frees it.
}

int main(int argc, char* argv[]){

  string ghpath            = "../grmfit/gh/data";
  string ghbase            = "hermite";
  size_t ghdim             = 2;
  size_t ghorder           = 12;
  bool   ghstep            = false;
  size_t threads           = 4;
  size_t limitvar          = SIZE_MAX;
  bool limitvar_rhat       = false;
  size_t t_bins            = 0;
  string fitnmout          = "";
  string fitnmin           = "";
  string siout             = "";
  string siin              = "";
  string testout           = "";
  string cvfile            = "";
  size_t ncovariates       =  1;
  bool renorm_cv           = false;
  string grmfile           = "";
  bool grm_bd              = false;
  string varfile           = "";
  size_t nvariants         = 0;
  bool use_vr              = false;
  string svar_ratio        = "";
  vector<double> var_ratio;             // one global variance ratio
  size_t loaded_vr_calibration_n=0;
  bool loaded_vr_calibration_n_known=false;
  double epsilon_vr        = 0.01;
  bool vr_diagnostics       = false;
  bool ap_audit = false;
  double ap_tau_start = 0.5;
  string vr_audit_tsv;
  string tau_if_audit_prefix;
  string tau_fd_audit_prefix;
  double tau_fd_step=0.001; // central FD half-step for cross-information
  string tau_cross_method="fd"; // normal test-time mode; "analytic" reproduces legacy tau cross-information
  bool tau_fd_cross_enabled=false; // active only for association testing
  bool tau_fd_fast_w1=true;         // FD computes first posterior moment only
  string hq_fd_audit_prefix;
  double hq_fd_step=1.e-4;
  bool fixed_tau_requested = false;
  double fixed_tau_value = 0.0;
  string vr_calibration_file = "";
  string vr_pool_output = "";
  size_t vr_pool_per_bin = 2000;
  uint64_t vr_seed = 2026;
  double spa_thresh        = numeric_limits<double>::max();
  bool cache_SI            = true;
  bool bnorm_ugamma        = false;
  bool utransform_G        = false;
  bool verbose             = false;
  const bool   b_last      = true;
  const bool   b_fullnr    = false;
  double epsilon           = 1.0e-5;
  bool absolute_thresh     = false;
  static struct option opts[]{
    {"vr-calibration-file", required_argument, NULL, 1001},
    {"make-vr-pool", required_argument, NULL, 1002},
    {"vr-pool-per-bin", required_argument, NULL, 1003}, // backward-compatible alias
    {"vr-pool-size", required_argument, NULL, 1007},
    {"vr-seed", required_argument, NULL, 1004},
    {"vr-diagnostics", no_argument, NULL, 1005},
    {"single-vr", no_argument, NULL, 1006}, // accepted, now the only mode
    {"ap-audit", no_argument, NULL, 1101},
    {"ap-tau-start", required_argument, NULL, 1102},
    {"vr-audit-tsv", required_argument, NULL, 1103},
    {"experimental-fixed-tau", required_argument, NULL, 1104},
    {"tau-if-audit-prefix", required_argument, NULL, 1105},
    {"tau-fd-audit-prefix", required_argument, NULL, 1106},
    {"tau-fd-step", required_argument, NULL, 1107},
    {"experimental-tau-fd-cross", no_argument, NULL, 1108}, // backward-compatible alias
    {"tau-cross", required_argument, NULL, 1113},
    {"tau-fd-fast-w1", no_argument, NULL, 1110},
    {"hq-fd-audit-prefix", required_argument, NULL, 1111},
    {"hq-fd-step", required_argument, NULL, 1112},
    {"help",             no_argument, NULL, 'h'},
    {"ghpath",     required_argument, NULL, 'g'},
    {"ghbase",     required_argument, NULL, 'G'},
    {"ndim",       required_argument, NULL, 'n'},
    {"order",      required_argument, NULL, 'o'},
    {"orderstep",        no_argument, NULL, 'O'},
    {"jthreads",   required_argument, NULL, 'j'},
    {"limitvar",   required_argument, NULL, 'l'},
    {"limitrhat",        no_argument, NULL, 'L'},
    {"tbins",      required_argument, NULL, 'T'},
    {"fitnmout",   required_argument, NULL, 'f'},
    {"fitnmin",    required_argument, NULL, 'F'},
    {"siout",      required_argument, NULL, 's'},
    {"siin",       required_argument, NULL, 'S'},
    {"testout",    required_argument, NULL, 't'},
    {"cvfile",     required_argument, NULL, 'c'},
    {"ncovariates",required_argument, NULL, 'C'},
    {"adjustcv",         no_argument, NULL, 'a'},
    {"absolutethresh",   no_argument, NULL, 'A'},
    {"grmfile",    required_argument, NULL, 'm'},
    {"grmbd",            no_argument, NULL, 'M'},
    {"varfile",    required_argument, NULL, 'b'},
    {"nvariants",  required_argument, NULL, 'B'},
    {"ratiovar",         no_argument, NULL, 'r'},
    {"ratioval",   required_argument, NULL, 'R'},
    {"epsilon",    required_argument, NULL, 'e'},
    {"epsilonvr",  required_argument, NULL, 'E'},
    {"spathresh",  required_argument, NULL, 'p'},
    {"donotcache",       no_argument, NULL, 'd'},
    {"ugammabnorm",      no_argument, NULL, 'u'},
    {"utransformG",      no_argument, NULL, 'U'},
    {"verbose",          no_argument, NULL, 'v'}};
  int opt = getopt_long(argc, argv,
			"hg:G:n:o:Oj:l:LT:f:F:s:S:t:c:C:aAm:Mb:B:rR:e:E:p:duUv",
			opts, NULL);
  while(opt != -1){
    switch(opt){
    case 1001: vr_calibration_file = string(optarg); break;
    case 1002: vr_pool_output = string(optarg); break;
    case 1003: vr_pool_per_bin = static_cast<size_t>(stoull(optarg)); break;
    case 1007: vr_pool_per_bin = static_cast<size_t>(stoull(optarg)); break;
    case 1004: vr_seed = static_cast<uint64_t>(stoull(optarg)); break;
    case 1005: vr_diagnostics = true; break;
    case 1006: break; // single VR is the default and sole supported mode
    case 1101: ap_audit = true; break;
    case 1102: ap_tau_start = stod(optarg); break;
    case 1103: vr_audit_tsv = string(optarg); break;
    case 1104: fixed_tau_requested = true; fixed_tau_value = stod(optarg); break;
    case 1105: tau_if_audit_prefix = string(optarg); break;
    case 1106: tau_fd_audit_prefix = string(optarg); break;
    case 1107: tau_fd_step = stod(optarg); break;
    case 1108: tau_cross_method="fd"; break; // legacy name for --tau-cross fd
    case 1113: tau_cross_method=string(optarg); break;
    case 1110: tau_fd_fast_w1=true; break;
    case 1111: hq_fd_audit_prefix=string(optarg); break;
    case 1112: hq_fd_step=stod(optarg); break;
    case 'h':
      cout << "options: "                              << endl
	   << "  -g path to GH data"                         << endl
	   << "  -n dimension of GH approx"                  << endl
	   << "  -o order     of GH approx"                  << endl
	   << "  -O step GH order"                           << endl
	   << "  -j number of threads"                       << endl
	   << "  -l limit number variants to read"           << endl
	   << "  -L stop reading variants once r-hat found"  << endl
	   << "  -T parameter for rebinning times"           << endl
	   << "  -f null model fit output"                   << endl
	   << "  -F null model fit input"                    << endl
	   << "  -s score/information output"                << endl
	   << "  -S score/information input"                 << endl
	   << "  -t score test output"                       << endl
	   << "  -c covariate csv file"                      << endl
	   << "  -C number of covariates"                    << endl
	   << "  -a adjust mean/sigma of covariates"         << endl
	   << "  -m grm hdf5 file"                    << endl
	   << "  -M grm block diagonal"               << endl
	   << "  -b bgen to hdf5 variant file"               << endl
	   << "  -B number of variants in csv file"          << endl
	   << "  -r use variance ratio for scores"           << endl
           << "  -R FILE load calibrated global VR (tagged HDF5 required for FD mode)" << endl
	   << "  -e set epsilon for null model"              << endl
	   << "  -E epsilon for variance ratio"              << endl
           << "  --vr-diagnostics print score/variance diagnostics" << endl
           << "  --tau-cross {fd|analytic}  tau cross-information method (default fd for score tests)" << endl
           << "  --tau-fd-step FLOAT    central-difference half-step (default 0.001)" << endl
           << "  --vr-pool-size N        calibration marker pool size (default 2000)" << endl
           << "  --make-vr-pool FILE    create a shuffled single-range VR pool" << endl
           << "  --vr-calibration-file FILE  read a shuffled VR calibration pool" << endl
           << "  --vr-seed UINT         random seed for creating a pool (default 2026)" << endl
           << "  (one global VR over MAF [0.001,0.5]; use -r to estimate/use it)" << endl
    << "  -p use SPA when abs(score) >= value"         << endl
	   << "  -d do not cache Stheta and Igamma"          << endl
	   << "  -v verbose mode"                            << endl;
      return 0;
    case 'g': ghpath      = string(optarg); break;
    case 'G': ghbase      = string(optarg); break;
    case 'n': ghdim       =   atoi(optarg); break;
    case 'o': ghorder     =   atoi(optarg); break;
    case 'O': ghstep      =true; ghorder=4; break;
    case 'j': threads     =   atoi(optarg); break;
    case 'l': limitvar    =   atoi(optarg); break;
    case 'L': limitvar_rhat=          true; break;
    case 'T': t_bins      =   atoi(optarg); break;
    case 'f': fitnmout    = string(optarg); break;
    case 'F': fitnmin     = string(optarg); break;
    case 's': siout       = string(optarg); break;
    case 'S': siin        = string(optarg); break;
    case 't': testout     = string(optarg); break;
    case 'c': cvfile      = string(optarg); break;
    case 'C': ncovariates =   atoi(optarg); break;
    case 'a': renorm_cv   =           true; break;
    case 'A': absolute_thresh =       true; break;
    case 'm': grmfile     = string(optarg); break;
    case 'M': grm_bd      =           true; break;
    case 'b': varfile     = string(optarg); break;
    case 'B': nvariants   =   atoi(optarg); break;
    case 'r': use_vr      =           true; break;
    case 'R': svar_ratio  = string(optarg); break;
    case 'e': epsilon     =   stod(optarg); break;
    case 'E': epsilon_vr  =   stod(optarg); break;
    case 'p': spa_thresh  =   stod(optarg); break;
    case 'd': cache_SI    =          false; break;
    case 'u': bnorm_ugamma=           true; break;
    case 'U': utransform_G=           true; break;
    case 'v': verbose     =           true; break;
    default:
      cout << "unrecoginized option -" << string(""+opt) << endl;
      return 1;
    }
    opt = getopt_long(argc, argv,
		      "hg:G:n:o:Oj:l:LT:f:F:s:S:t:c:C:aAm:Mb:B:rR:e:E:p:duUv",
		      opts, NULL);
  }

  if(!vr_pool_output.empty()){
    if(varfile.empty()) throw runtime_error("--make-vr-pool requires -b SOURCE_SPARSE_HDF5");
    const vector<double> pool_edges = {0.001, 0.5};
    BuildRandomVRPool(varfile,vr_pool_output,vr_pool_per_bin,vr_seed,pool_edges);
    return 0;
  }
  if(!vr_calibration_file.empty() && !use_vr)
    throw runtime_error("--vr-calibration-file requires -r");
  if(!std::isfinite(ap_tau_start) || ap_tau_start<=0.0)
    throw runtime_error("--ap-tau-start must be positive and finite");
  if(!vr_audit_tsv.empty()){
#ifdef __CUDA__
    throw runtime_error("--vr-audit-tsv is CPU-only; use the CPU GADO build");
#endif
    if(use_vr || !svar_ratio.empty())
      throw runtime_error("--vr-audit-tsv computes exact variance for every SNP: remove -r/-R");
    if(varfile.empty() && nvariants == 0)
      throw runtime_error("--vr-audit-tsv requires -b GENOTYPE_HDF5 or -B NVARIANTS");
    if(testout.empty() || limitvar==SIZE_MAX)
      throw runtime_error("--vr-audit-tsv requires -t OUTPUT and -l N (limit exact SNPs)");
    if(!vr_calibration_file.empty() || !vr_pool_output.empty())
      throw runtime_error("Audit TSV uses standard -b or -B input, not shuffled VR pool");
    if(limitvar_rhat)
      throw runtime_error("--vr-audit-tsv: do not use -L; held-out SNPs require full -l N");
  }
  // The null-model fitting path is intentionally unaffected by this option.
  // FD tau cross-information is the default only when running a score test (-t).
  // Null-model-only (-f/-F/-s/-S) workflows do not run finite differences.
  if(tau_cross_method!="fd" && tau_cross_method!="analytic")
    throw runtime_error("--tau-cross must be 'fd' or 'analytic'");
  if(!std::isfinite(tau_fd_step) || tau_fd_step<=0.0)
    throw runtime_error("--tau-fd-step must be positive and finite");
  tau_fd_cross_enabled = (tau_cross_method=="fd" && !testout.empty() && !fixed_tau_requested);
  tau_fd_fast_w1 = true;
  if(!tau_fd_audit_prefix.empty()){
    if(testout.empty())
      throw runtime_error("--tau-fd-audit-prefix requires -t OUTPUT_HDF5");
  }
  if(!hq_fd_audit_prefix.empty()){
    if(!std::isfinite(hq_fd_step) || hq_fd_step<=0.0 || hq_fd_step>=0.1)
      throw runtime_error("--hq-fd-step must lie between 0 and 0.1");
    if(siin.empty() || varfile.empty() || testout.empty() || use_vr ||
       !svar_ratio.empty() || limitvar!=1 || limitvar_rhat)
      throw runtime_error("H/q FD audit requires -S, -b, -t, -l 1; no -r/-R/-L");
#ifndef __CUDA__
    // Validation-only audit is implemented on CPU.
#else
    throw runtime_error("H/q FD audit is CPU-only");
#endif
  }
  // Method options have already been validated and are recorded in the
  // HDF5 output where applicable. Suppress the lengthy method banner.
  // Retain the FD -R compatibility check without changing its condition.
  if(tau_fd_cross_enabled && !svar_ratio.empty() && !use_vr)
    throw runtime_error("FD -R precomputed ratios require -r");
  if(!testout.empty() && !limitvar_rhat &&
     spa_thresh==numeric_limits<double>::max())
    cout << "[GADO_NOTE] SPA is disabled unless -p THRESHOLD is specified (e.g., -p 2)." << endl;
  if(fixed_tau_requested){
    if(!std::isfinite(fixed_tau_value) || fixed_tau_value<=0.0)
      throw runtime_error("--experimental-fixed-tau must be positive and finite");
    if(!fitnmin.empty() || !fitnmout.empty() || !siin.empty() || !siout.empty())
      throw runtime_error("Fixed-tau experiment must refit directly; do not use -f/-F/-s/-S");
    if(vr_audit_tsv.empty())
      throw runtime_error("Experimental fixed tau requires --vr-audit-tsv output");
    cout << "EXPERIMENT ONLY: treating tau=" << fixed_tau_value
         << " as KNOWN, omitting its estimator influence."
         << " NOT a standard GWAS null model." << endl;
  }
  assert(cvfile  != "");
  assert(grmfile != "");
  assert(fitnmin == "" || fitnmout == "");
  assert(siin    == "" || siout    == "");

  PMatrix<double>*     T = new PMatrix<double>("T", cvfile,
					       vector<size_t>({2}));
  if(t_bins > 0){
    size_t nbins = T->BinValues(t_bins);
    cout << "condensed observation times into " << nbins
	 << " variable width bins" << endl;
  }
  PMatrix<double>* delta = new PMatrix<double>("delta", cvfile,
					       vector<size_t>({1}));
  vector<size_t> cvindex(ncovariates);
  iota(cvindex.begin(), cvindex.end(), 3);
  PMatrix<double>*     X = new PMatrix<double>("X", cvfile, cvindex);
  if(renorm_cv){
    for(size_t i=0; i<X->GetNCols(); i++){
      bool bincol = true;
      for(size_t j=0; j<X->GetNRows(); j++){
	double x = X->GetVal(j, i);
	if(x != (double) ((int) x) || x < 0.0 || x > 1.0){
	  bincol = false;
	  break;
	}
      }
      if(!bincol) X->RenormalizeCol(i);
    }
  }
  H5::H5File grmh5(grmfile.c_str(), H5F_ACC_RDONLY);
  BDMatrix<double>*  psi;
  if(!grm_bd){
    psi = new BDMatrix<double>("psi");
    PMatrix<double>* grm = new PMatrix<double>("sGRM", grmh5);
    grmh5.close();
    psi->AppendBlock(grm);
    delete grm;
    cout << "Appended single dense block to GRM with " << psi->GetNBlocks()
	 << " blocks after diagonalization" << endl;
  }
  else{
    psi = new BDMatrix<double>("psi", "block_", grmh5, true, false);
    grmh5.close();
  }
  // Sort and permute the data without routine startup timing logs.
  psi->SortBlocks();
  vector<size_t> permutation = psi->GetIndexPermutation();
  T->PermuteRows(permutation);
  delta->PermuteRows(permutation);
  X->PermuteRows(permutation);

  GHAModel* model = new GHAModel(ghpath, ghbase, ghdim, ghorder, ghstep,
				 T, delta, X, psi);
  // One global VR is the only supported calibration mode.
  model->SetVarRatioMAFEdges(vector<double>({0.001, 0.5}));
  model->SetEpsilon(epsilon);
  model->SetAbsoluteThresh(absolute_thresh);
  model->SetBLast(b_last);
  model->SetBFullNR(b_fullnr);
  model->SetVerbose(verbose);
  model->SetGHStepIterConv(50);
  model->SetUseVarRatio(use_vr);
  model->SetVarRatioEpsilon(epsilon_vr);
  model->SetVarRatioDiagnostics(vr_diagnostics);
  model->SetAPDiagnostics(ap_audit);
  model->SetAPTauSeed(ap_tau_start);
  model->SetVRAuditEnabled(!vr_audit_tsv.empty());
  model->SetTauFDCross(tau_fd_cross_enabled);
  model->SetTauFDFastW1(tau_fd_fast_w1);
  model->SetHQFDAuditEnabled(!hq_fd_audit_prefix.empty());
  if(fixed_tau_requested) model->SetExperimentalFixedTau(fixed_tau_value);
  const double checked_vr_epsilon=model->GetVarRatioEpsilon();
  if(!std::isfinite(checked_vr_epsilon) || checked_vr_epsilon<=0.0 || checked_vr_epsilon>=1.0)
    throw runtime_error("Invalid GADO model VR epsilon: possible shared-library/ABI mismatch");
  model->SetSPAThreshold(spa_thresh);
  model->SetCacheSthetaIgamma(cache_SI);
  model->SetBNormUgamma(bnorm_ugamma);
  model->SetCentralizeG(!utransform_G);
  model->SetUnitaryTransformG(utransform_G);

  // ------------------------------------------------------------------
  // Single global MAF calibration range; the edges themselves come
  // from GHAModel, here they are only reported.
  // ------------------------------------------------------------------
  const size_t nvrbins = model->GetNVarRatioBins();
  if(nvrbins==0 || nvrbins>32)
    throw runtime_error("Invalid GADO number of VR bins: possible shared-library/ABI mismatch");
  const vector<double>& vr_edges = model->GetVarRatioMAFEdges();
  if(vr_edges.size()!=nvrbins+1)
    throw runtime_error("Inconsistent VR bin edges: possible shared-library/ABI mismatch");
  if(nvrbins!=1) throw runtime_error("Final GADO requires exactly one global VR");

  // ------------------------------------------------------------------
  // Load an optional global rhat (tagged, calibrated HDF5 in FD mode).
  // VR metadata will be printed once, after the file has been validated.
  // ------------------------------------------------------------------
  if(svar_ratio.size() > 0){
    char* end_test;
    double v = strtod(svar_ratio.c_str(), &end_test);
    if(end_test != svar_ratio.c_str() && *end_test == '\0'){
      if(tau_fd_cross_enabled)
        throw runtime_error("FD tau-cross requires tagged HDF5 -R; a scalar ratio has no method provenance");
      var_ratio.assign(nvrbins, v);
    }
    else{
      H5::H5File fvr(svar_ratio.c_str(), H5F_ACC_RDONLY);
      vector<double> stored_count;
      if(ReadH5DoubleVector(fvr,"vr_calibration_n",stored_count) &&
         stored_count.size()==1 && std::isfinite(stored_count[0]) &&
         stored_count[0]>=0.0){
        loaded_vr_calibration_n=static_cast<size_t>(stored_count[0]);
        loaded_vr_calibration_n_known=true;
      }else{
        vector<double> stored_samples;
        if(ReadH5DoubleVector(fvr,"var_ratio_estimates_0",stored_samples)){
          loaded_vr_calibration_n=stored_samples.size();
          loaded_vr_calibration_n_known=true;
        }
      }
      if(tau_fd_cross_enabled){
        vector<double> method_tag, stored_fd_step;
        if(!ReadH5DoubleVector(fvr, "fd_tau_cross_method_version", method_tag) ||
           method_tag.size()!=1 || method_tag[0]!=1.0)
          throw runtime_error("FD tau cross -R requires fd_tau_cross_method_version=1; recalibrate or verify and tag existing FD ratios");
        if(!ReadH5DoubleVector(fvr, "fd_tau_half_step", stored_fd_step) ||
           stored_fd_step.size()!=1 || !std::isfinite(stored_fd_step[0]) ||
           std::fabs(stored_fd_step[0]-tau_fd_step)>1e-12)
          throw runtime_error("FD tau cross -R requires matching fd_tau_half_step; do not mix VR calibrations");
      }
      vector<double> stored_edges;
      if(ReadH5DoubleVector(fvr, "vr_maf_edges", stored_edges)){
        if(stored_edges.size()!=vr_edges.size())
          throw runtime_error("-R VR MAF edges are incompatible with the single global VR mode");
        for(size_t j=0;j<stored_edges.size();++j)
          if(std::fabs(stored_edges[j]-vr_edges[j])>1e-12)
            throw runtime_error("-R VR MAF edges do not match single global VR mode");
      }
      if(!ReadH5DoubleVector(fvr, "var_ratio_bins", var_ratio)){
	// fall back to the old single-value format
	if(!ReadH5DoubleVector(fvr, "var_ratio", var_ratio)){
	  cerr << "ERROR: no variance ratio found in " << svar_ratio << endl;
	  return 1;
	}
	if(var_ratio.size() == 1 && nvrbins > 1){
	  cout << "WARNING: " << svar_ratio << " holds a single variance "
	       << "ratio, applying it to every MAF bin" << endl;
	  var_ratio.assign(nvrbins, var_ratio[0]);
	}
      }
      fvr.close();
    }
    if(var_ratio.size() != nvrbins){
      cerr << "ERROR: expected one global variance ratio, got "
           << var_ratio.size() << endl;
      return 1;
    }
    if(tau_fd_cross_enabled && !svar_ratio.empty()){
      for(size_t b=0;b<nvrbins;++b)
        if(!std::isfinite(var_ratio[b]) || var_ratio[b]<=0.0)
          throw runtime_error("FD tau cross -R requires positive finite VR in every MAF bin");
    }
    for(size_t b=0; b<nvrbins; b++){
      if(var_ratio[b] > 0.0) model->SetVarRatio(b, var_ratio[b]);
      else cout << "WARNING: MAF bin " << b << " has no usable variance "
		<< "ratio, it will be evaluated exactly" << endl;
    }
  }

  // Print a concise global-VR summary; preserve all FD provenance checks
  // and HDF5 metadata without echoing those details in routine logs.
  if(use_vr && !testout.empty()){
    cout << "[GADO_VR] GLOBAL_MEDIAN; MAF=["
         << Log4(vr_edges.front()) << "," << Log4(vr_edges.back()) << "]";
    if(!svar_ratio.empty() && model->AllVarRatioFound()){
      cout << "; rhat=" << Log4(model->GetVarRatio(0));
    }else{
      cout << "; epsilon=" << Log4(checked_vr_epsilon);
    }
    cout << endl;
  }

  openblas_set_num_threads(threads);
  #ifdef __CUDA__
    cublasHandle_t chandle;
    cublasCreate_v2(&chandle);
    model->SetCublasHandle(chandle);
  #endif
  if(fitnmin == ""){
    if(siin == "") model->Solve(100000, 2000);
    if(fitnmout != "") model->WriteNullModelHDF5(fitnmout);
  }
  else model->ReadNullModelHDF5(fitnmin);
  if(siin == ""){
    if(siout != ""){
      model->ComputeScoreInfo();
      model->WriteScoreInfoHDF5(siout);
    }
    else if(testout != ""){
      model->ComputeScoreInfo();
    }
  }
  else model->ReadScoreInfoHDF5(siin);
  if(!tau_if_audit_prefix.empty())
    model->WriteTauIFAudit(tau_if_audit_prefix);
  if(tau_fd_cross_enabled || !tau_fd_audit_prefix.empty()){
    model->SetTauFDStep(tau_fd_step);
    // With a tagged, supplied -R, nearly all GWAS variants use the VR path.
    // Avoid the expensive FD initialization per chromosome; GHAModel prepares
    // it lazily if a variant requires the exact-variance fallback.
    if(!tau_fd_audit_prefix.empty() || !model->AllVarRatioFound())
      model->PrepareTauFD(tau_fd_audit_prefix);
    else if(verbose)
      cout << "[GADO_TAU_CROSS] FD preparation deferred until an exact-variance fallback is needed." << endl;
  }
  if(!vr_calibration_file.empty() && !model->AllVarRatioFound()){
    CalibrateFromVRPool(vr_calibration_file,model,permutation,T->GetNRows());
  }
  // Calibration-only mode: -L with --vr-calibration-file produces a
  // compact file that can be loaded later via -R, without scanning GWAS data.
  if(!vr_calibration_file.empty() && limitvar_rhat){
    if(testout.empty()) throw runtime_error("Calibration-only mode requires -t OUTPUT_HDF5");
    H5::H5File fout(testout.c_str(),H5F_ACC_TRUNC);
    vector<double> values(nvrbins);
    for(size_t b=0;b<nvrbins;++b) values[b]=model->GetVarRatio(b);
    WriteH5DoubleVector(fout,"var_ratio_bins",values);
    WriteH5DoubleVector(fout,"vr_maf_edges",vr_edges);
    WriteH5DoubleVector(fout,"vr_nbins",vector<double>{static_cast<double>(nvrbins)});
    if(tau_fd_cross_enabled){
      WriteH5DoubleVector(fout,"fd_tau_cross_method_version",vector<double>{1.0});
      WriteH5DoubleVector(fout,"fd_tau_half_step",vector<double>{tau_fd_step});
    }
    // Legacy readers may use the scalar 'var_ratio'; both now represent one VR.
    WriteH5DoubleVector(fout,"var_ratio",values);
    WriteH5DoubleVector(fout,"vr_calibration_n",vector<double>{
      static_cast<double>(model->GetVarRatioEstimates(0).size())});
    for(size_t b=0;b<nvrbins;++b)
      WriteH5DoubleVector(fout,"var_ratio_estimates_"+to_string(b),model->GetVarRatioEstimates(b));
    fout.close();
    cout << "Wrote VR-only result to " << testout << endl;
    delete model;
    delete T;
    delete delta;
    delete X;
    delete psi;
    return 0;
  }
  if(testout != "" || nvariants > 0){
    PMatrix<double>* G = new PMatrix<double>("G", T->GetNRows(),
					     T->GetNCols(), false);
    PMatrix<double>* G_all = NULL;
    SparseDosageReader* reader = NULL;
    vector<double> pval;
    if(varfile != ""){
      if(testout != ""){
        // The first reader only queries sample/variant metadata and is then
        // destroyed without reading any variant. Suppress its misleading
        // zero-variant rate (and any duplicate constructor progress output).
        // This is a one-time, single-threaded initialization phase.
        const auto reader_init_start = steady_clock::now();
        std::string reader_init_log;
        {
          ScopedReaderLogCapture capture;
          reader = new SparseDosageReader(varfile);
          if(T->GetNRows() < reader->GetNInd()){
            permutation.resize(reader->GetNInd());
            iota(permutation.begin()+psi->GetNRows(),
                 permutation.end(), psi->GetNRows());
            G_all = new PMatrix<double>("G_all", reader->GetNInd(),
                                        T->GetNCols(), false);
          }
          delete reader;
          reader = new SparseDosageReader(varfile, permutation);
          reader_init_log = capture.str();
        }
        ReplayNonProgressReaderMessages(reader_init_log);
        const double reader_init_s =
          duration<double>(steady_clock::now()-reader_init_start).count();
        cout << "SparseDosageReader: " << reader->GetNVariants()
             << " variants available; initialization: "
             << Log4(reader_init_s) << " s" << endl;
        pval.resize(std::min(reader->GetNVariants(), limitvar));
      }
    }
    else{
      vector<size_t> gindex(nvariants);
      iota(gindex.begin(), gindex.end(), 3+ncovariates);
      G_all = new PMatrix<double>("G_all", cvfile, gindex);
      G_all->PermuteRows(permutation);
      pval.resize(nvariants);
      limitvar = nvariants;
    }
    vector<double> pval_gc(pval.size());
    vector<double> score_stat(pval.size());
    vector<double> Ugamma(pval.size());
    vector<double> Ugamma_var(pval.size());
    vector<double> maf(pval.size());
    vector<double> mac(pval.size());
    vector<double> vrbin(pval.size());   // MAF bin used for each variant
    std::ofstream audit_out;
    if(!vr_audit_tsv.empty()){
      audit_out.open(vr_audit_tsv.c_str());
      if(!audit_out) throw runtime_error("Cannot open VR audit TSV: "+vr_audit_tsv);
      audit_out << std::setprecision(17);
      audit_out << "variant_index\tmode\ttau\tmaf_bin\tMAF\tMAC"
                << "\tUgamma\tV_raw\tV_cross\tV_correction\tV_exact"
                << "\tV_no_tau_correction\tD_working\tratio"
                << "\tD_block_proxy_HEURISTIC\tratio_block_proxy_HEURISTIC"
                << "\talgebra_error\torthogonality_max"
                << "\tdominant_block_fraction\tdominant_block_size"
                << "\tdominant_block_events"
                << "\tdominant_block_id\th_tau"
                << "\tsum_Asc2_scaled\tsum_v_Asc_scaled"
                << "\ttau_cross_term\ttau_squared_term"
                << "\ttau_decomposition_error"
                << "\ttop_tau_block_id\ttop_tau_block_size"
                << "\ttop_tau_block_events\ttop_tau_block_fraction"
                << "\tI_gamma_tau_analytic\tI_gamma_tau_fd"
                << "\ttau_projection_analytic\ttau_projection_fd_c"
                << "\th_tau_fd_c\tI_gamma_tau_gap\th_tau_fd_c_gap\n";
    }
    size_t ivar = 0;
    // Count time spent in NextVariant(), not the much longer score test.
    // Genotypes and the association computations remain unchanged.
    std::chrono::nanoseconds total_reader_time(0);
    auto read_next_variant = [&]() -> bool {
      const auto read_start = steady_clock::now();
      const bool success = reader->NextVariant(G_all ? G_all : G);
      total_reader_time += duration_cast<nanoseconds>(steady_clock::now()-read_start);
      return success;
    };
    auto print_reader_progress = [&]() {
      if(!reader || ivar==0) return; // Never print 0 ns/variant.
      cout << "SparseDosageReader: read " << ivar << "/"
           << reader->GetNVariants() << " variants; avg="
           << Log4(static_cast<double>(total_reader_time.count()) / ivar)
           << " ns/variant; read_time="
           << Log4(total_reader_time.count() * 1e-9) << " s" << endl;
    };
    while((reader ? read_next_variant() : true) &&
	  ivar < limitvar){
      if(reader){ if(G_all) G_all->GetBlock(0, 0, G); }
      else G_all->GetBlock(0, ivar, G);
      model->SetG(G);
      const bool doprint = (ivar % 1000 == 0);
      if(doprint)
        cout << "variant " << ivar << endl;
      model->ComputeScoreTest(doprint);
      if(!hq_fd_audit_prefix.empty() && ivar==0){
        model->RunHQFDAudit(hq_fd_audit_prefix,hq_fd_step);
      }
      score_stat[ivar]      = model->GetScoreStat();
      Ugamma[ivar]          = model->GetUgamma();
      Ugamma_var[ivar]      = model->GetUgammaVar();
      pval[ivar]            = model->GetPVal();
      maf[ivar]             = model->GetMAF();
      mac[ivar]             = model->GetMAC();
      vrbin[ivar]           = (double) model->GetCurrentVarRatioBin();
      if(audit_out.is_open()){
        const GHAModel::VRAuditRecord& a = model->GetVRAuditRecord();
        if(!a.valid) throw runtime_error("Audit expected exact variance for every SNP");
        audit_out << ivar << '\t'
                  << (fixed_tau_requested ? "FIXED_TAU_KNOWN" :
                     (tau_fd_cross_enabled ? "ESTIMATED_TAU_FD_CROSS" :
                                               "ESTIMATED_TAU")) << '\t'
                  << model->GetTau() << '\t' << model->GetCurrentVarRatioBin()
                  << '\t' << model->GetMAF() << '\t' << model->GetMAC()
                  << '\t' << model->GetUgamma()
                  << '\t' << a.raw << '\t' << a.cross
                  << '\t' << a.correction << '\t' << a.exact
                  << '\t' << a.no_tau_correction << '\t' << a.working
                  << '\t' << a.ratio << '\t' << a.block_proxy
                  << '\t' << a.ratio_block_proxy
                  << '\t' << a.algebra_error
                  << '\t' << a.orthogonality_max
                  << '\t' << a.dominant_block_fraction
                  << '\t' << a.dominant_block_size
                  << '\t' << a.dominant_block_events
                  << '\t' << a.dominant_block_id
                  << '\t' << a.tau_adjust_coeff
                  << '\t' << a.sum_asc2_scaled
                  << '\t' << a.sum_v_asc_scaled
                  << '\t' << a.tau_cross_term
                  << '\t' << a.tau_squared_term
                  << '\t' << a.tau_decomposition_error
                  << '\t' << a.top_tau_block_id
                  << '\t' << a.top_tau_block_size
                  << '\t' << a.top_tau_block_events
                  << '\t' << a.top_tau_block_fraction
                  << '\t' << a.I_gamma_tau_analytic
                  << '\t' << a.I_gamma_tau_fd
                  << '\t' << a.tau_projection_analytic
                  << '\t' << a.tau_projection_fd_c
                  << '\t' << a.h_tau_fd_c
                  << '\t' << a.I_gamma_tau_gap
                  << '\t' << a.h_tau_fd_c_gap << '\n';
      }
      ivar ++;
      if(ivar % 1000 == 0 && reader) print_reader_progress();
      if(limitvar_rhat && vr_calibration_file.empty() &&
         model->GetVarRatioConverged()) break;
    }

    vector<double> vr_out(nvrbins);
    for(size_t b=0; b<nvrbins; b++) vr_out[b] = model->GetVarRatio(b);

    // the legacy scalar argument keeps the common variant bin
    GHAWriteBGENScoreTestHDF5(model, testout, reader, pval,
                                score_stat,
                                Ugamma, Ugamma_var, maf, mac,
                                vr_out.empty() ? 0.0 : vr_out.back());

    // append the per-bin variance ratios and the per-variant bin index
    if(testout != ""){
      H5::H5File fout(testout.c_str(), H5F_ACC_RDWR);
      WriteH5DoubleVector(fout, "var_ratio_bins", vr_out);
      const size_t current_calibration_n=model->GetVarRatioEstimates(0).size();
      if(current_calibration_n>0 || loaded_vr_calibration_n_known)
        WriteH5DoubleVector(fout, "vr_calibration_n", vector<double>{
          static_cast<double>(current_calibration_n>0 ? current_calibration_n : loaded_vr_calibration_n)});
      WriteH5DoubleVector(fout, "vr_maf_edges",   vr_edges);
      WriteH5DoubleVector(fout, "vr_nbins",       vector<double>{static_cast<double>(nvrbins)});
      WriteH5DoubleVector(fout, "maf_bin",        vrbin);
      // The provenance marker avoids accidentally mixing analytic and FD VR.
      if(tau_fd_cross_enabled){
        WriteH5DoubleVector(fout, "fd_tau_cross_method_version", vector<double>{1.0});
        WriteH5DoubleVector(fout, "fd_tau_half_step", vector<double>{tau_fd_step});
      }
      // GHAIO writes the concatenation of every bin under
      // "var_ratio_estimates", which mixes the bins together; keep the
      // per-bin samples separately so the medians stay meaningful
      for(size_t b=0; b<nvrbins; b++)
	WriteH5DoubleVector(fout, "var_ratio_estimates_"+to_string(b),
			    model->GetVarRatioEstimates(b));
      fout.close();
    }

    // A loaded -R ratio was already reported above.  For a newly
    // calibrated ratio, print its result once after all tested variants.
    if(use_vr && svar_ratio.empty()){
      cout << "[GADO_VR] calibration_result: rhat=" << Log4(vr_out[0])
           << (model->GetVarRatioConverged(0) ? "; converged=yes" : "; converged=no")
           << endl;
    }
    
    // Report the final partial batch without forcing a 0/variant rate.
    if(reader && ivar>0 && ivar%1000!=0) print_reader_progress();
    if(reader){
      std::string reader_close_log;
      {
        ScopedReaderLogCapture capture;
        delete reader; // Keep original object lifetime and cleanup.
        reader_close_log = capture.str();
      }
      ReplayNonProgressReaderMessages(reader_close_log);
    }
    if(G_all)  delete G_all;
  }

  delete model;
  delete T;
  delete delta;
  delete X;
  delete psi;

  return 0;

}
