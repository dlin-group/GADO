#include "PMatrix.hh"
#include <H5Cpp.h>
#include <TFile.h>
#include <TRInterface.h>
#include <iostream>

using namespace std;

int main(){

  // a random matrix to write and read in
  PMatrix<double>* a = new PMatrix<double>("a", 10, 10, false);
  a->Random();
  cout << "Initial random matrix:" << endl;
  a->Print();

  // test writing and reading to HDF5 format
  H5::H5File* hdf5_file = new H5::H5File("data/hdf5_test.root", H5F_ACC_TRUNC);
  a->WriteHDF5(*hdf5_file, "a");
  delete hdf5_file;
  hdf5_file = new H5::H5File("data/hdf5_test.root", H5F_ACC_RDONLY);
  PMatrix<double>* b = new PMatrix<double>("a", *hdf5_file);
  cout << "Matrix read from HDF5 file:" << endl;
  b->Print();
  delete b;
  delete hdf5_file;
  
  // test writing and reading to ROOT format
  TFile* root_file = new TFile("data/root_test.root", "recreate");
  a->WriteROOT(root_file, "a");
  root_file->Close();
  delete root_file;
  root_file = TFile::Open("data/root_test.root");
  b = new PMatrix<double>("a", root_file);
  cout << "Matrix read from ROOT file:" << endl;
  b->Print();
  delete b;
  root_file->Close();

  // test writing and reading to Rdata format
  ROOT::R::TRInterface& R = ROOT::R::TRInterface::Instance();
  a->WriteRdata("data/r_test.Rdata", "a", R);
  b = new PMatrix<double>("a", "data/r_test.Rdata", R);
  cout << "Matrix read from Rdata file:" << endl;
  b->Print();
  delete b;
  
  delete a;
  
  return 0;
}
