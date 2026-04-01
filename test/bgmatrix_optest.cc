#include "PMatrix.hh"
#include "GMatrix.hh"
#include "BGMatrix.hh"
#include <cublas_v2.h>

using namespace std;

int main(){

  PMatrix<double>* a1 = new PMatrix<double>("a1", 1, 1, true);
  a1->SetVal(0, 0.99);  
  PMatrix<double>* a2 = new PMatrix<double>("a2", 2, 2, true);
  a2->SetVal(0, 1.0); a2->SetVal(1, 0.9); a2->SetVal(2, 0.1);a2->SetVal(3, 1.0);
  PMatrix<double>* a3 = new PMatrix<double>("a3", 3, 3, false);
  a3->SetVal(1, 0.25); a3->SetVal(2,  0.75);
  a3->SetVal(3, 0.25); a3->SetVal(5, -0.25);
  a3->SetVal(6, 0.75); a3->SetVal(7, -0.25);

  cublasHandle_t handle;
  cublasCreate_v2(&handle);
  
  BGMatrix<double>* b = new BGMatrix<double>("b", handle);
  b->AppendBlock(a1);
  b->AppendBlock(a2);
  b->AppendBlock(a1);
  b->AppendBlock(a3);
  b->AppendBlock(a2);
  //b->AppendBlock(a1);
  //b->AppendBlock(a3);
  b->CloseToAppend();
  b->Print();

  a1->Random();
  a2->Random();
  a3->Random();
  
  BGMatrix<double>* c = new BGMatrix<double>("c", handle);
  c->AppendBlock(a1);
  c->AppendBlock(a2);
  c->AppendBlock(a1);
  c->AppendBlock(a3);
  c->AppendBlock(a2);
  //c->AppendBlock(a1);
  //c->AppendBlock(a3);
  c->CloseToAppend();
  c->Print();

  b->ChangeDevLock(false);
  c->ChangeDevLock(false);

  BGMatrix<double>* d = (*b) + (*c);
  d->ChangeDevLock(true);
  d->Print();

  BGMatrix<double>* e = (*b) * (*c);
  e->ChangeDevLock(true);
  e->Print();

  BGMatrix<double>* f = new BGMatrix<double>("f", *c);
  c->Inverse(f);
  f->ChangeDevLock(true);
  f->Print();
  f->ChangeDevLock(false);
  BGMatrix<double>* g = new BGMatrix<double>("g", *c);
  c->RightMult(*f, 1.0, g, 0.0);
  g->ChangeDevLock(true);
  g->Print();

  GMatrix<double>* h = new GMatrix<double>("h", b->GetNRows(), 1, false,handle);
  h->Random();
  h->CopyToDevice();
  GMatrix<double>* i = new GMatrix<double>("i", b->GetNRows(), 1, false,handle);
  i->CopyToDevice();
  b->RightMult(*h, 1.0, i, 0.0);
  i->ReadFromDevice();
  h->Print(25, 3);
  i->Print(25, 3);
  
  delete a1;
  delete a2;
  delete a3;
  delete b;
  delete c;
  delete d;
  delete e;
  delete f;
  delete g;
  cublasDestroy_v2(handle);
  
  return 0;
}
