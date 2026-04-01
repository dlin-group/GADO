#include "PMatrix.hh"
#include "BDMatrix.hh"

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
  
  BDMatrix<double>* b = new BDMatrix<double>("b");
  b->AppendBlock(a1);
  b->AppendBlock(a2);
  b->AppendBlock(a1);
  b->AppendBlock(a3);
  b->AppendBlock(a2);
  //b->AppendBlock(a1);
  //b->AppendBlock(a3);
  b->SortBlocks();
  b->Print();

  a1->Random();
  a2->Random();
  a3->Random();
  BDMatrix<double>* c = new BDMatrix<double>("c");
  c->AppendBlock(a1);
  c->AppendBlock(a2);
  c->AppendBlock(a1);
  c->AppendBlock(a3);
  c->AppendBlock(a2);
  //c->AppendBlock(a1);
  //c->AppendBlock(a3);
  c->SortBlocks();
  c->Print();

  BDMatrix<double>* d = (*b) + (*c);
  d->Print();

  BDMatrix<double>* e = (*b) * (*c);
  e->Print();

  BDMatrix<double>* f = c->Inverse();
  e->Zero();
  c->RightMult(*f, e);
  f->Print();
  e->Print();

  PMatrix<double>* g = new PMatrix<double>("g", b->GetNRows(), 1, false);
  g->Random();
  PMatrix<double>* h = new PMatrix<double>("h", b->GetNRows(), 1, false);
  b->RightMult(*g, h);
  g->Print(25, 3);
  h->Print(25, 3);

  delete a1;
  delete a2;
  delete a3;
  delete b;
  delete c;
  delete d;
  delete e;
  delete f;
  delete g;
  delete h;
  
  return 0;
}
