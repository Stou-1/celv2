// A fork-join helper for the encoder's short parallel steps.
#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "core/profile.h"

namespace celv2 {

// fn(k) for k in [0, n) on threads kept for the process (hardware_concurrency - 1 of them, started
// on first use) and this one, each k once, in no fixed pairing of k to thread, so the work must
// not rely on the k's running at the same time. Starting 48 threads
// several times a frame cost more than 1080p's filter searches themselves. It may be called from
// inside a task: the caller works through its own k's, and idle threads help. The first exception
// from fn is rethrown once every k has run.
class ThreadPool {
 public:
  static ThreadPool& get();
  template <class F>
  void run(int n, const F& fn) {
    if (n <= 0) return;
    if (n == 1) return fn(0);
    Job j;
    j.n = n;
    j.stage = current_stage();  // CELV2_PROFILE: the helpers' time counts as the caller's stage
    j.fn = [&fn](int k) { fn(k); };
    {
      std::lock_guard<std::mutex> lock(mu_);
      jobs_.push_back(&j);
    }
    work_.notify_all();
    work(j);
    std::unique_lock<std::mutex> lock(mu_);
    done_.wait(lock, [&] { return j.done == j.n && j.helpers == 0; });
    jobs_.erase(std::find(jobs_.begin(), jobs_.end(), &j));
    lock.unlock();
    if (j.err) std::rethrow_exception(j.err);
  }
  ~ThreadPool();

 private:
  struct Job {
    std::function<void(int)> fn;
    int n = 0;
    std::atomic<int> next{0}, done{0};
    int helpers = 0;  // pool threads inside work() on it (under mu_)
    Stage stage = Stage::None;
    std::exception_ptr err;
    std::mutex err_mu;
  };
  ThreadPool();
  void work(Job& j);
  void loop();
  std::mutex mu_;
  std::condition_variable work_, done_;
  std::vector<Job*> jobs_;
  std::vector<std::thread> threads_;
  bool stop_ = false;
};

template <class F>
void parallel(int n, const F& fn) {
  ThreadPool::get().run(n, fn);
}

}  // namespace celv2
