#include <climits>
#include <cstddef>
#include <cstdint>

#ifndef CONSTANTS_H
#define CONSTANTS_H

#define MAX_BUF_SIZE 1024
enum time_stages {
  MASTER_WORKER,
  ALL_REDUCE,
  BCAST,
  OVERALL_FILE_LOAD,
  OVERALL_DISTRIBUTE_FUNCTION,
  OVERALL_TOTAL
};

typedef uint64_t unit_t;
#define MPI_UNIT_T MPI_UINT64_T

struct sets_t {
  size_t numRows;
  size_t numTumor;
  size_t numNormal;
  size_t numCols;
  unit_t *data;
};
#endif

// ############MACROS####################
#ifdef ENABLE_TIMING
#define START_TIMING(var) double var##_start = MPI_Wtime();
#define END_TIMING(var, accumulated_time)                                      \
  do {                                                                         \
    double var##_end = MPI_Wtime();                                            \
    accumulated_time += var##_end - var##_start;                               \
  } while (0)
#else
#define START_TIMING(var)
#define END_TIMING(var, accumulated_time)
#endif

#define BITS_PER_UNIT (sizeof(unit_t) * CHAR_BIT)

#define SHIFT_LEFT(x, n) (((n) >= BITS_PER_UNIT) ? (unit_t)0 : ((x) << (n)))
#define SHIFT_RIGHT(x, n) (((n) >= BITS_PER_UNIT) ? (unit_t)0 : ((x) >> (n)))

#define ALL_BITS_SET (~(unit_t)0)

#define UNITS_FOR_BITS(N) (((N) + BITS_PER_UNIT - 1) / BITS_PER_UNIT)

#define MASK_RANGE_UNITT(firstBit, lastBit)                                    \
  (SHIFT_RIGHT(ALL_BITS_SET, (firstBit)) &                                     \
   SHIFT_LEFT(ALL_BITS_SET, (BITS_PER_UNIT - 1 - (lastBit))))

#define EXTRACT_TUMOR_BITS(OUT, TABLE, GENE)                                   \
  do {                                                                         \
    const size_t numTumor = (TABLE).numTumor;                                  \
    const size_t rowOffset = ((size_t)(GENE)) * (TABLE).numCols;               \
    const size_t rowLastBit = rowOffset + (TABLE).numCols - 1;                 \
    const size_t tumorStart = rowOffset;                                       \
    const size_t tumorEnd = rowOffset + numTumor - 1;                          \
    const size_t actualEnd = (tumorEnd <= rowLastBit ? tumorEnd : rowLastBit); \
    const size_t neededUnits = UNITS_FOR_BITS(numTumor);                       \
    std::memset((OUT), 0, neededUnits * sizeof(unit_t));                       \
    size_t firstBlockIndex = tumorStart / BITS_PER_UNIT;                       \
    size_t firstBlockOffset = tumorStart % BITS_PER_UNIT;                      \
    size_t lastBlockIndex = actualEnd / BITS_PER_UNIT;                         \
    size_t lastBlockOffset = actualEnd % BITS_PER_UNIT;                        \
    size_t localBitPos = 0;                                                    \
    if (firstBlockOffset != 0 || (firstBlockIndex == lastBlockIndex)) {        \
      unit_t srcVal = (TABLE).data[firstBlockIndex];                           \
      size_t endOff = (firstBlockIndex == lastBlockIndex)                      \
                          ? lastBlockOffset                                    \
                          : (BITS_PER_UNIT - 1);                               \
      unit_t mask = MASK_RANGE_UNITT(firstBlockOffset, endOff);                \
      unit_t extracted = srcVal & mask;                                        \
      extracted = SHIFT_RIGHT(extracted, firstBlockOffset);                    \
      size_t usedBits = (endOff - firstBlockOffset + 1);                       \
      (OUT)[localBitPos / BITS_PER_UNIT] |=                                    \
          (extracted << (localBitPos % BITS_PER_UNIT));                        \
      localBitPos += usedBits;                                                 \
      if (firstBlockIndex == lastBlockIndex)                                   \
        break;                                                                 \
      firstBlockIndex++;                                                       \
    }                                                                          \
    while (firstBlockIndex < lastBlockIndex) {                                 \
      unit_t srcVal = (TABLE).data[firstBlockIndex];                           \
      (OUT)[localBitPos / BITS_PER_UNIT] |=                                    \
          (srcVal << (localBitPos % BITS_PER_UNIT));                           \
      localBitPos += BITS_PER_UNIT;                                            \
      firstBlockIndex++;                                                       \
    }                                                                          \
    if (firstBlockIndex == lastBlockIndex) {                                   \
      unit_t srcVal = (TABLE).data[lastBlockIndex];                            \
      unit_t mask = MASK_RANGE_UNITT(0, lastBlockOffset);                      \
      unit_t extracted = srcVal & mask;                                        \
      (OUT)[localBitPos / BITS_PER_UNIT] |=                                    \
          (extracted << (localBitPos % BITS_PER_UNIT));                        \
      localBitPos += (lastBlockOffset + 1);                                    \
    }                                                                          \
  } while (0)

#define EXTRACT_NORMAL_BITS(OUT, TABLE, GENE)                                  \
  do {                                                                         \
    const size_t numNormal = (TABLE).numNormal;                                \
    const size_t numTumor = (TABLE).numTumor;                                  \
    const size_t rowOffset = ((size_t)(GENE)) * (TABLE).numCols;               \
    const size_t actualEnd = rowOffset + (TABLE).numCols - 1;                  \
    const size_t normalEnd = actualEnd;                                        \
    const size_t normalStart = rowOffset + numTumor;                           \
    const size_t neededUnits = UNITS_FOR_BITS(numNormal);                      \
    std::memset((OUT), 0, neededUnits * sizeof(unit_t));                       \
    size_t firstBlockIndex = normalStart / BITS_PER_UNIT;                      \
    size_t firstBlockOffset = normalStart % BITS_PER_UNIT;                     \
    size_t lastBlockIndex = actualEnd / BITS_PER_UNIT;                         \
    size_t lastBlockOffset = actualEnd % BITS_PER_UNIT;                        \
    size_t localBitPos = 0;                                                    \
    if (firstBlockOffset != 0 || (firstBlockIndex == lastBlockIndex)) {        \
      unit_t srcVal = (TABLE).data[firstBlockIndex];                           \
      size_t endOff = (firstBlockIndex == lastBlockIndex)                      \
                          ? lastBlockOffset                                    \
                          : (BITS_PER_UNIT - 1);                               \
      unit_t mask = MASK_RANGE_UNITT(firstBlockOffset, endOff);                \
      unit_t extracted = srcVal & mask;                                        \
      extracted = SHIFT_RIGHT(extracted, firstBlockOffset);                    \
      size_t usedBits = (endOff - firstBlockOffset + 1);                       \
      (OUT)[localBitPos / BITS_PER_UNIT] |=                                    \
          (extracted << (localBitPos % BITS_PER_UNIT));                        \
      localBitPos += usedBits;                                                 \
      if (firstBlockIndex == lastBlockIndex)                                   \
        break;                                                                 \
      firstBlockIndex++;                                                       \
    }                                                                          \
    while (firstBlockIndex < lastBlockIndex) {                                 \
      unit_t srcVal = (TABLE).data[firstBlockIndex];                           \
      (OUT)[localBitPos / BITS_PER_UNIT] |=                                    \
          (srcVal << (localBitPos % BITS_PER_UNIT));                           \
      localBitPos += BITS_PER_UNIT;                                            \
      firstBlockIndex++;                                                       \
    }                                                                          \
    if (firstBlockIndex == lastBlockIndex) {                                   \
      unit_t srcVal = (TABLE).data[lastBlockIndex];                            \
      unit_t mask = MASK_RANGE_UNITT(0, lastBlockOffset);                      \
      unit_t extracted = srcVal & mask;                                        \
      (OUT)[localBitPos / BITS_PER_UNIT] |=                                    \
          (extracted << (localBitPos % BITS_PER_UNIT));                        \
      localBitPos += (lastBlockOffset + 1);                                    \
    }                                                                          \
  } while (0)

#define INTERSECT_BUFFERS(OUT, A, B, UNITS)                                    \
  do {                                                                         \
    for (size_t _u = 0; _u < (UNITS); _u++) {                                  \
      (OUT)[_u] = (A)[_u] & (B)[_u];                                           \
    }                                                                          \
  } while (0)

// ############MACROS####################
