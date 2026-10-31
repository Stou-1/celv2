#include "core/threads.h"

namespace celv2 {

ThreadPool& ThreadPool::get() {
  static ThreadPool pool;
  return pool;
}

ThreadPool::ThreadPool() {
  const int n = std::max(1, static_cast<int>(std::thread::hardware_concurrency())) - 1;
  try {
    for (int i = 0; i < n; i++) threads_.emplace_back([this] { loop(); });
  } catch (...) {  // fewer threads than asked: the callers do the rest
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  work_.notify_all();
  for (std::thread& t : threads_) t.join();
}

void ThreadPool::work(Job& j) {
  for (int k; (k = j.next++) < j.n;) {
    try {
      j.fn(k);
    } catch (...) {
      std::lock_guard<std::mutex> lock(j.err_mu);
      if (!j.err) j.err = std::current_exception();
    }
    if (++j.done == j.n) {
      std::lock_guard<std::mutex> lock(mu_);  // the waiter checks under mu_: no lost wake-up
      done_.notify_all();
    }
  }
}

void ThreadPool::loop() {
  std::unique_lock<std::mutex> lock(mu_);
  for (;;) {
    Job* j = nullptr;
    work_.wait(lock, [&] {
      if (stop_) return true;
      for (auto it = jobs_.rbegin(); it != jobs_.rend(); ++it)  // the newest first: nested jobs get help
        if ((*it)->next < (*it)->n) return j = *it, true;
      return false;
    });
    if (!j) return;
    j->helpers++;
    lock.unlock();
    work(*j);
    lock.lock();
    j->helpers--;
    done_.notify_all();
  }
}

}  // namespace celv2
