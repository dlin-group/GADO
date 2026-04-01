#include "ThreadPool.hh"

using namespace std;

ThreadPool::ThreadPool(size_t num_threads){
  stop = false;
  idle.resize(num_threads, true);
  for(size_t i=0; i<num_threads; ++i){
    threads.emplace_back([this, i]{
      while(true){
	pair<function<void(uint32_t)>, uint32_t> task;
	{
	  unique_lock<mutex> lock(queue_mutex);
	  cond_var.wait(lock, [this, i]{
	    return !tasks.empty() || stop;
	  });
	  if(stop && tasks.empty()) return;
	  idle[i] = false;
	  task = move(tasks.front());
	  tasks.pop();
	}
	task.first(task.second);
	idle[i] = true;
      } 
    });
  }
}

ThreadPool::~ThreadPool(){
  {
    unique_lock<mutex> lock(queue_mutex);
    stop = true;
  }
  cond_var.notify_all();
  for(auto& thread : threads) thread.join();
}

void ThreadPool::enqueue(pair<function<void(uint32_t)>, uint32_t> task){
  {
    unique_lock<std::mutex> lock(queue_mutex);
    tasks.emplace(move(task)); 
  } 
  cond_var.notify_one(); 
}


void ThreadPool::join(){
  while(any_of(idle.begin(), idle.end(), [](bool i){ return !i; }))
    this_thread::sleep_for(chrono::nanoseconds(100));
}
