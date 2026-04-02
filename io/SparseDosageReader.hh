#ifndef __SparseDosageReader__
#define __SparseDosageReader__

#include "PMatrix.hh"
#include <H5Cpp.h>
#include <vector>
#include <string>

using namespace std;
using namespace H5;

class SparseDosageReader{
public:

  SparseDosageReader(string infname,
		     vector<size_t> idperm=vector<size_t>(),
		     double maxsize_gb=(1<<30)/1.0e9);
  virtual ~SparseDosageReader();

  virtual size_t GetNInd()           const { return ids.size();               }
  virtual size_t GetNVariants()      const { return rsids.size();             }
  virtual size_t GetCurrentVariant() const { return ivariant;                 }
  virtual double GetTotalTimeSec()   const { return time_ns/1.0e9;            }
  virtual double GetnsPerVariant()   const
  { return (ivariant==0 ? 0.0 : time_ns/(double)ivariant); }

  virtual bool NextVariant(PMatrix<double>* G);
  virtual void PrintTime();
  virtual void WriteHeader(H5File* outfile, string base="");
  
  virtual const vector<size_t>& GetIDs()         const { return ids;         }
  virtual const vector<string>& GetRSIDs()       const { return rsids;       }
  virtual const vector<string>& GetChromosomes() const { return chromosomes; }
  virtual const vector<string>& GetAllele0()     const { return allele0;     }
  virtual const vector<string>& GetAllele1()     const { return allele1;     }
  virtual const vector<uint32_t>& GetPositions() const { return positions;   }

protected:

  bool NextChunk();

  size_t time_ns;
  size_t ivariant;
  
  H5File* infile;
  DataSet dset;
  DataSpace fspace;
  DataSpace mspace;
  hsize_t fdim[1];
  hsize_t mdim[1];
  hsize_t start[1];
  hsize_t offset[1];

  vector<size_t> ids;
  vector<size_t> id_perm;
  vector<string> rsids;
  vector<string> chromosomes;
  vector<string> allele0;
  vector<string> allele1;
  vector<uint32_t> positions;
  vector<uint32_t> data;
  vector<uint32_t>::iterator data_it;

  uint16_t delta;
  uint32_t dval;
  uint32_t uval;
  double   val;
  size_t index;
  size_t lindex;
  size_t vindex;
  size_t sindex;
  uint32_t dbits;
  uint32_t dmax;
  uint32_t dmask;
  uint32_t vbits;
  uint32_t vmax;
  uint32_t vmask;

};

#endif
