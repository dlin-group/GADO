#include <string>
#include "BMatrix.hh"
#include <iostream>
#include <fstream>
#include <assert.h>
#include "genfile/bgen/bgen.hpp"
#include "genfile/bgen/View.hpp"


using namespace std;
using namespace genfile;

struct ProbSetter{
  typedef vector<vector< double> > Data ;
  ProbSetter(Data* result) : m_result(result), m_sample_i(0){}

  void initialise(size_t number_of_samples, size_t number_of_alleles){
    m_result->clear();
    m_result->resize(number_of_samples);
  }
	
  /* If present with this signature, called once after initialise()
     to set the minimum and maximum ploidy and numbers of probabilities
     among samples in the data. This enables us to set up storage for
     the data ahead of time. */
  void set_min_max_ploidy(uint32_t min_ploidy, uint32_t max_ploidy,
			  uint32_t min_entries, uint32_t max_entries){
    for(size_t i = 0; i < m_result->size(); ++i)
      m_result->at(i).reserve(max_entries);
  }
	
  // Called once per sample to determine whether we want data for this sample
  bool set_sample(size_t i){
    m_sample_i = i ;
    // Yes, here we want info for all samples.
    return true ;
  }
	
  // Called once per sample to set the number of probabilities that are present
  void set_number_of_entries(size_t ploidy, size_t number_of_entries,
			     OrderType order_type,
			     ValueType value_type){
    assert(value_type == eProbability);
    m_result->at(m_sample_i).resize(number_of_entries);
    m_entry_i = 0;
  }
  
  // Called once for each genotype (or haplotype) probability per sample.
  void set_value(uint32_t, double value){
    m_result->at(m_sample_i).at(m_entry_i++) = value;
  }

  // Ditto, but called if data is missing for this sample.
  void set_value(uint32_t, MissingValue value) {
    // Here we encode missing probabilities with -1
    m_result->at(m_sample_i).at(m_entry_i++) = -1;
  }

  // If present with this signature, called once after all data has been set.
  void finalise() {
    // nothing to do in this implementation.
  }

private:
  Data* m_result;
  size_t m_sample_i;
  size_t m_entry_i;
};


/* BgenParser is a thin wrapper around the core functions in bgen.hpp.
   This class tracks file state and handles passing the right callbacks. */
struct BgenParser {
  BgenParser(string const& filename) : m_filename(filename),
				       m_state(e_NotOpen),
				       m_have_sample_ids(false){
    // Open the stream
    m_stream.reset(new ifstream(filename, ifstream::binary));
    if(!*m_stream) throw invalid_argument(filename);
    m_state = e_Open;

    // Read the offset, header, and sample IDs if present.
    bgen::read_offset(*m_stream, &m_offset);
    bgen::read_header_block(*m_stream, &m_context);
    if(m_context.flags & bgen::e_SampleIdentifiers){
      bgen::read_sample_identifier_block(*m_stream, m_context,
					 [this](string id){
					   m_sample_ids.push_back(id); });
      m_have_sample_ids = true ;
    }
		
    // Jump to the first variant data block.
    m_stream->seekg(m_offset + 4);
    stream_count = 0;

    // We keep track of state
    m_state = e_ReadyForVariant;
  }

  ostream& summarise(ostream& o) const{
    o << "BgenParser: bgen file ("
      << (m_context.flags & bgen::e_Layout2 ? "v1.2 layout" : "v1.1 layout")
      << ", "
      << (m_context.flags & bgen::e_CompressedSNPBlocks ?
	  "compressed" : "uncompressed") << ")" << " with " 
      << m_context.number_of_samples << " "
      << ( m_have_sample_ids ? "named" : "anonymous" ) << " samples and "
      << m_context.number_of_variants << " variants.\n";
    return o;
  }

  size_t number_of_samples() const{
    return m_context.number_of_samples;
  }

  size_t number_of_variants() const{
    return m_context.number_of_variants;
  }

  size_t get_stream_count() const{
    return stream_count;
  }

  // Report the sample IDs in the file using the given setter object
  // (If there are no sample IDs in the file, we report a dummy identifier).
  template<typename Setter>
  void get_sample_ids(Setter setter){
    if(m_have_sample_ids){
      for(size_t i = 0; i<m_context.number_of_samples; ++i)
	setter(m_sample_ids[i]);
    }
    else{
      for(size_t i = 0; i<m_context.number_of_samples; ++i)
	setter("(unknown_sample_" + to_string(i+1) + ")");
    }
  }

  /* Attempt to read identifying information about a variant from the bgen
     file, returning it in the given fields. If this method returns true,
     data was successfully read, and it should be safe to call read_probs()
     or ignore_probs(). If this method returns false, data was not
     successfully read indicating the end of the file. */
  bool read_variant(string* chromosome, uint32_t* position,
		    string* rsid, vector<string>* alleles){
    assert(m_state == e_ReadyForVariant);
    string SNPID ; // read but ignored in this toy implementation
    stream_count ++;
    if(bgen::read_snp_identifying_data(*m_stream, m_context,
				       &SNPID, rsid, chromosome, position,
				       [&alleles](size_t n){
					 alleles->resize(n);},
				       [&alleles]( size_t i,
						   string const& allele){
					 alleles->at(i) = allele;})){
      m_state = e_ReadyForProbs;
      return true ;
    }
    else return false ;
  }
	
  /* Read genotype probability data for the SNP just read using read_variant()
     After calling this method it should be safe to call read_variant() to
     fetchthe next variant from the file. */
  void read_probs( vector<vector<double> >* probs){
    assert(m_state == e_ReadyForProbs);
    ProbSetter setter(probs);
    bgen::read_and_parse_genotype_data_block<ProbSetter>(*m_stream, m_context,
							 setter,
							 &m_buffer1,
							 &m_buffer2);
    m_state = e_ReadyForVariant;
  }

  /* Ignore genotype probability data for the SNP just read using read_variant()
     After calling this method it should be safe to call read_variant()
     to fetch the next variant from the file. */
  void ignore_probs() {
    bgen::ignore_genotype_data_block(*m_stream, m_context);
    m_state = e_ReadyForVariant;
  }

private:
  string const m_filename;
  unique_ptr<istream> m_stream;
  size_t stream_count;
  
  // bgen::Context object holds information from the header block
  bgen::Context m_context;
  
  // offset byte from top of bgen file.
  uint32_t m_offset;
  
  /* We keep track of our state in the file.
     Not strictly necessary for this implentation but makes it clear that
     calls must be read_variant() followed by read_probs() (or ignore_probs())
     repeatedly. */
  enum State { e_NotOpen = 0, e_Open = 1, e_ReadyForVariant = 2,
	       e_ReadyForProbs = 3, eComplete = 4 };
  State m_state;

  // If the BGEN file contains samples ids, they will be read here.
  bool m_have_sample_ids;
  vector<string> m_sample_ids;
	
  // Buffers, these are used as working space by bgen implementation.
  vector<byte_t> m_buffer1, m_buffer2;
};


bool NextSparseBGENDosages(BgenParser& parser, size_t& offset, size_t& lindex,
			   vector<string>& chromosomes,
			   vector<uint32_t>& positions,
			   vector<string>& allele0, vector<string>& allele1,
			   vector<string>& rsids,
			   vector<uint32_t>& data, size_t max_size);
