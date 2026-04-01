#ifndef __ThreadPool__
#define __ThreadPool__

#include <condition_variable> 
#include <functional> 
#include <iostream> 
#include <mutex> 
#include <queue> 
#include <thread>
#include <utility>

using namespace std; 

class ThreadPool{
public: 
  ThreadPool(size_t num_threads=thread::hardware_concurrency());
  ~ThreadPool();

  void enqueue(pair<function<void(uint32_t)>, uint32_t> task);
  void join();
  size_t ntasks() const { return tasks.size(); }
  size_t nthreads() const { return threads.size(); }
  
private: 
  vector<thread> threads;
  queue<pair<function<void(uint32_t)>, uint32_t> > tasks;
  mutex queue_mutex;
  condition_variable cond_var;
  bool stop;
  vector<bool> idle;
};

#endif
