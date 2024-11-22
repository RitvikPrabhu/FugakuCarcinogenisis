#include <omp.h>
#include <random>
#include <stdio.h>

#define N 24
#define REP 4

int main(int argc, char *argv[]) {
  printf("Hello\n");

  std::default_random_engine rng;
  std::uniform_real_distribution<> dist(0, 2 * N);
  double a[N];

  for (size_t i = 0; i < N; ++i) {
    a[i] = dist(rng);
  }

  double time = omp_get_wtime();
  double maxval;
  size_t maxidx;
#pragma omp parallel
  {
    double maxval_prv;
    size_t maxidx_prv;
#pragma omp for nowait
    for (size_t i = 0; i < N; ++i) {
      int num_threads = omp_get_num_threads();
      int tid = omp_get_thread_num();
      if (i == 0)
        printf("%d/%d\n", tid, num_threads);
      double val = a[i];
      for (int t = 0; t < REP; t++) {
        val += (a[i] + 2 * (0.5 - (t & 1)) * t);
      }
      val /= REP;

      if (val > maxval_prv) {
        maxval_prv = a[i];
        maxidx_prv = i;
      }
    }
#pragma omp critical
    {
      if (maxval_prv > maxval) {
        maxval = maxval_prv;
        maxidx = maxidx_prv;
      }
    }
  }
  printf("max[i] = %f @ %d\n", maxval, maxidx);
  printf("time: %f\n", omp_get_wtime() - time);
  return 0;
}
