// Code based on Ulrich Drepper. What Every Programmer Should Know About Memory. 2007.
#include <error.h>
#include <pthread.h>
#include <stdlib.h>

#include <cassert>
#include <iostream>

#define MemaAssert(expr, msg)      \
  if (!static_cast<bool>(expr)) {  \
    std::cerr << msg << std::endl; \
    return 1;                      \
  }                                \
  static_assert(true, "End call of macro with a semicolon")

#define OP_COUNT (atomic ? 10000000 : 500000000)

static int atomic;
static unsigned thread_count;
static unsigned size_factor_per_thread;
static pthread_barrier_t barrier;

static void* workload(void* arg) {
  long* ptr = reinterpret_cast<long*>(arg);
  if (atomic) {
    for (int op_idx = 0; op_idx < OP_COUNT; ++op_idx) {
      __sync_add_and_fetch(ptr, 1);
    }
    return NULL;
  }

  for (int op_idx = 0; op_idx < OP_COUNT; ++op_idx) {
    *ptr += 1;
    asm volatile("" : : "m"(*ptr));
  }
  return NULL;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    size_factor_per_thread = 0;
  } else {
    size_factor_per_thread = atol(argv[1]);
  }

  if (argc < 3) {
    thread_count = 2;
  } else {
    thread_count = atol(argv[2]) ?: 1;
  }

  if (argc < 4) {
    atomic = 1;
  } else {
    atomic = atol(argv[3]);
  }

  std::cout << "thread count: " << thread_count << ", size_factor_per_thread: " << size_factor_per_thread
            << ", atomic: " << atomic << std::endl;

  pthread_barrier_init(&barrier, NULL, thread_count);
  void* addr;
  const auto success = posix_memalign(&addr, 64, (thread_count * size_factor_per_thread ?: 1) * sizeof(long));
  MemaAssert(success == 0, "posic_memalign failed");

  long* data = reinterpret_cast<long*>(addr);
  pthread_t threads[thread_count];
  pthread_attr_t attribute;
  pthread_attr_init(&attribute);
  cpu_set_t cpu_set;

  for (unsigned thread_idx = 1; thread_idx < thread_count; ++thread_idx) {
    // set affinity in thread attribute
    CPU_ZERO(&cpu_set);
    CPU_SET(thread_idx, &cpu_set);
    const auto setaffinity_success = pthread_attr_setaffinity_np(&attribute, sizeof(cpu_set), &cpu_set);
    MemaAssert(setaffinity_success == 0, "pthread_attr_setaffinity_np failed");

    // zero thread data
    data[thread_idx * size_factor_per_thread] = 0;

    // create child thread with configured thread attribute and run workload with thread specific data address
    const auto create_thread_success =
        pthread_create(&threads[thread_idx], &attribute, workload, &data[thread_idx * size_factor_per_thread]);
    MemaAssert(create_thread_success == 0, "create_thread_success failed");
  }

  // set affinity for main thread
  CPU_ZERO(&cpu_set);
  CPU_SET(0, &cpu_set);
  const auto setaffinity_success = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);
  MemaAssert(setaffinity_success == 0, "pthread_attr_setaffinity_np failed");

  // zero main thread data
  data[0] = 0;

  // run workload with main thread specific data address
  workload(&data[0]);

  if ((size_factor_per_thread == 0 && data[0] != thread_count * OP_COUNT) ||
      (size_factor_per_thread != 0 && data[0] != OP_COUNT)) {
    error(1, 0, "data[0] wrong: %ld instead of %d", data[0],
          size_factor_per_thread == 0 ? thread_count * OP_COUNT : OP_COUNT);
  }

  for (unsigned thread_idx = 1; thread_idx < thread_count; ++thread_idx) {
    pthread_join(threads[thread_idx], NULL);
    if (size_factor_per_thread != 0 && data[thread_idx * size_factor_per_thread] != OP_COUNT) {
      error(1, 0, "data[%u] wrong: %ld instead of %d", thread_idx, data[thread_idx * size_factor_per_thread], OP_COUNT);
    }
  }
  return 0;
}