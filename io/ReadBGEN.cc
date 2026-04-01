#include "ReadBGEN.hh"
#include "BMatrix.hh"
#include <assert.h>

using namespace std;
using namespace genfile;

bool NextSparseBGENDosages(BgenParser& parser, size_t& offset, size_t& lindex,
			   vector<string>& chromosomes,
			   vector<uint32_t>& positions,
			   vector<string>& allele0, vector<string>& allele1,
			   vector<string>& rsids,
			   vector<uint32_t>& data, size_t max_size){
  cout << (int)(offset/(double)parser.number_of_variants()*100) << "%" << endl;
  string chromosome;
  uint32_t position;
  string rsid;
  vector<string> alleles;
  vector<vector<double> > probs;
  size_t vcount = offset;
  size_t scount = 0;
  size_t delta = 0;
  uint32_t uval = 0;
  double val = 0.0;
  const uint32_t dbits = 10, dmax = (1 << dbits);//, dmask = 0x3FF;
  const uint32_t vbits = 20, vmax = (1 << vbits), vmask = 0xFFFFF;
  while(parser.read_variant(&chromosome, &position, &rsid, &alleles)){
    assert(alleles.size() >= 2);
    parser.read_probs(&probs);
    chromosomes.push_back(chromosome);
    positions.push_back(position);
    allele0.push_back(alleles[0]);
    allele1.push_back(alleles[1]);
    rsids.push_back(rsid);
    for(scount=0; scount<probs.size(); scount++){
      assert(probs[scount].size() == 3);
      val = probs[scount][1] + 2*probs[scount][2];
      delta = vcount*parser.number_of_samples() + scount - lindex;
      if(val >= (1.0 / dmax) || delta == dmax-1){
	assert(val >= 0.0 && val <= 2.0);
	assert(delta < dmax);
	data.push_back((uint32_t) delta);
	if(val >= 1.0 && val < 2.0){
	  data.back() += dmax;
	  val -= 1.0;
	}
	else if(val == 2.0){
	  data.back() += dmax << 1;
	  val = 0.0;
	}
	uval = (uint32_t) (val * vmax);
	data.back() += (uval & vmask) << (dbits+2);
	/*uint32_t utest = 0;
	double test = 0.0;
	uint16_t dtest = (uint16_t) (data.back() & dmask);
	if(data.back() &  dmax)       test += 1.0;
	if(data.back() & (dmax << 1)) test += 2.0;
	utest = (val == 0.0) ? 0 : (data.back() &
				    (vmask << (dbits+2))) >> (dbits+2);
	test += utest / (double) vmax;
	cout << val << " " << test << endl;
	cout << abs(test - (int)(test) - val) << " " << 1./vmax << endl;
	cout << uval << " " << utest << " " << delta << " "
	     << (uint32_t) dtest << endl;
	getchar();
	assert(uval == utest && delta == (size_t) dtest &&
	abs(test - (int)(test) - val) < 1.0/vmax);*/
	lindex = vcount*parser.number_of_samples() + scount;
      }
    }
    vcount ++;
    if(data.size() >= max_size){
      offset = vcount;
      return true;
    }
  }
  return false;
}

