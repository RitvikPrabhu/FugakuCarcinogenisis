#include <climits>
#include <cstddef>
#include <cstdint>

#ifndef CARCUTILS_H
#define CARCUTILS_H

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
  unit_t *tumorData;
  unit_t *normalData;
};

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

#define ALL_BITS_SET (~(unit_t)0)

#define UNITS_FOR_BITS(N) (((N) + BITS_PER_UNIT - 1) / BITS_PER_UNIT)
/**
inline bool is_empty(const uint64_t *buf, size_t units) {
  for (size_t i = 0; i < units; i++) {
    if (buf[i] != 0ULL)
      return false;
  }
  return true;
}

inline int bitCollection_size(const uint64_t *buf, size_t units) {
  int count = 0;
  for (size_t i = 0; i < units; i++) {
    count += __builtin_popcountll(buf[i]);
  }
  return count;
}

inline void inplace_intersect_tumor(unit_t *scratch, const sets_t &table,
                                    size_t gene) {

  const size_t numTumor = table.numTumor;
  const size_t rowOffset = gene * table.numCols;
  const size_t rowEnd = rowOffset + table.numCols - 1;

  size_t startBit = rowOffset;
  size_t endBit = rowOffset + numTumor - 1;
  if (endBit > rowEnd)
    endBit = rowEnd;

  size_t dstPos = 0;

  size_t firstBlockIndex = startBit / BITS_PER_UNIT;
  size_t firstBlockOffset = startBit % BITS_PER_UNIT;
  size_t lastBlockIndex = endBit / BITS_PER_UNIT;
  size_t lastBlockOffset = endBit % BITS_PER_UNIT;

  while (true) {
    size_t endOff = (firstBlockIndex == lastBlockIndex ? lastBlockOffset
                                                       : (BITS_PER_UNIT - 1));

    unit_t blockVal = table.data[firstBlockIndex];

    if (firstBlockOffset != 0 || firstBlockIndex == lastBlockIndex) {
      uint64_t leftMask = ~(((unit_t)1 << firstBlockOffset) - (unit_t)1);

      blockVal &= leftMask;
    }
    blockVal >>= firstBlockOffset;

    size_t usedBits = (endOff - firstBlockOffset + 1);

    size_t dstBlock = dstPos / BITS_PER_UNIT;
    size_t dstOff = dstPos % BITS_PER_UNIT;

    uint64_t shiftedVal = (blockVal << dstOff);

    size_t blockEnd = dstOff + usedBits - 1;
    if (blockEnd >= BITS_PER_UNIT)
      blockEnd = BITS_PER_UNIT - 1;

    unit_t localMask = 0;
    if (dstOff <= blockEnd) {
      if (dstOff == 0 && blockEnd == 63) {
        localMask = ALL_BITS_SET;
      } else {
        unit_t rm = ((blockEnd >= (BITS_PER_UNIT - 1))
                         ? ALL_BITS_SET
                         : ((unit_t)1 << (blockEnd + 1)) - (unit_t)1);
        uint64_t lm = ((dstOff == 0) ? ALL_BITS_SET
                                     : ~(((unit_t)1 << dstOff) - (unit_t)1));
        localMask = rm & lm;
      }
    }

    unit_t oldVal = scratch[dstBlock];
    unit_t oldInside = oldVal & localMask;
    unit_t newInside = oldInside & (shiftedVal & localMask);
    scratch[dstBlock] = (oldVal & ~localMask) | newInside;

    dstPos += usedBits;
    if (firstBlockIndex == lastBlockIndex)
      break;

    firstBlockIndex++;
    firstBlockOffset = 0;
  }
}

inline void load_first_tumor(unit_t *scratch, const sets_t &table,
                             size_t gene) {

  const size_t numTumor = table.numTumor;
  const size_t rowOffset = gene * table.numCols;
  const size_t rowEnd = rowOffset + table.numCols - 1;

  size_t startBit = rowOffset;
  size_t endBit = rowOffset + numTumor - 1;
  if (endBit > rowEnd)
    endBit = rowEnd;

  size_t dstPos = 0;

  size_t firstBlockIndex = startBit / BITS_PER_UNIT;
  size_t firstBlockOffset = startBit % BITS_PER_UNIT;
  size_t lastBlockIndex = endBit / BITS_PER_UNIT;
  size_t lastBlockOffset = endBit % BITS_PER_UNIT;

  while (true) {
    size_t endOff = (firstBlockIndex == lastBlockIndex ? lastBlockOffset
                                                       : (BITS_PER_UNIT - 1));

    unit_t blockVal = table.data[firstBlockIndex];

    if (firstBlockOffset != 0 || firstBlockIndex == lastBlockIndex) {
      uint64_t leftMask = ~(((unit_t)1 << firstBlockOffset) - (unit_t)1);

      blockVal &= leftMask;
    }
    blockVal >>= firstBlockOffset;

    size_t usedBits = (endOff - firstBlockOffset + 1);

    size_t dstBlock = dstPos / BITS_PER_UNIT;
    size_t dstOff = dstPos % BITS_PER_UNIT;

    uint64_t shiftedVal = (blockVal << dstOff);

    size_t blockEnd = dstOff + usedBits - 1;
    if (blockEnd >= BITS_PER_UNIT)
      blockEnd = BITS_PER_UNIT - 1;

    unit_t localMask = 0;
    if (dstOff <= blockEnd) {
      if (dstOff == 0 && blockEnd == 63) {
        localMask = ALL_BITS_SET;
      } else {
        unit_t rm = ((blockEnd >= (BITS_PER_UNIT - 1))
                         ? ALL_BITS_SET
                         : ((unit_t)1 << (blockEnd + 1)) - (unit_t)1);
        uint64_t lm = ((dstOff == 0) ? ALL_BITS_SET
                                     : ~(((unit_t)1 << dstOff) - (unit_t)1));
        localMask = rm & lm;
      }
    }

    scratch[dstBlock] =
        (scratch[dstBlock] & ~localMask) | (shiftedVal & localMask);

    dstPos += usedBits;
    if (firstBlockIndex == lastBlockIndex)
      break;

    firstBlockIndex++;
    firstBlockOffset = 0;
  }
}

inline void inplace_intersect_normal(unit_t *scratch, const sets_t &table,
                                     size_t gene) {
  const size_t numTumor = table.numTumor;
  const size_t numCols = table.numCols;

  const size_t rowOffset = gene * numCols;
  size_t startBit = rowOffset + numTumor;
  size_t endBit = rowOffset + (numCols - 1);

  size_t dstPos = 0;

  size_t firstBlockIndex = startBit / BITS_PER_UNIT;
  size_t firstBlockOffset = startBit % BITS_PER_UNIT;
  size_t lastBlockIndex = endBit / BITS_PER_UNIT;
  size_t lastBlockOffset = endBit % BITS_PER_UNIT;

  while (true) {
    size_t endOff = (firstBlockIndex == lastBlockIndex) ? lastBlockOffset
                                                        : (BITS_PER_UNIT - 1);

    unit_t blockVal = table.data[firstBlockIndex];

    if (firstBlockOffset != 0 || (firstBlockIndex == lastBlockIndex)) {
      unit_t rightMask;
      if (endOff >= (BITS_PER_UNIT - 1)) {
        rightMask = ALL_BITS_SET;
      } else {
        rightMask = (((unit_t)1 << (endOff + 1)) - (unit_t)1);
      }
      blockVal &= rightMask;
    }

    blockVal >>= firstBlockOffset;

    size_t usedBits = (endOff - firstBlockOffset + 1);

    size_t dstBlock = dstPos / BITS_PER_UNIT;
    size_t dstOff = dstPos % BITS_PER_UNIT;

    unit_t shiftedVal = blockVal << dstOff;

    size_t blockEnd = dstOff + usedBits - 1;
    if (blockEnd >= BITS_PER_UNIT) {
      blockEnd = BITS_PER_UNIT - 1;
    }
    unit_t localMask = 0ULL;
    if (dstOff <= blockEnd) {
      unit_t rm;
      if (blockEnd >= (BITS_PER_UNIT - 1)) {
        rm = ALL_BITS_SET;
      } else {
        rm = (((unit_t)1 << (blockEnd + 1)) - (unit_t)1);
      }
      unit_t lm;
      if (dstOff == 0) {
        lm = ALL_BITS_SET;
      } else {
        lm = ~(((unit_t)1 << dstOff) - (unit_t)1);
      }
      localMask = (rm & lm);
    }

    unit_t oldVal = scratch[dstBlock];
    unit_t oldInside = (oldVal & localMask);
    uint64_t newInside = (oldInside & (shiftedVal & localMask));
    scratch[dstBlock] = (oldVal & ~localMask) | newInside;

    dstPos += usedBits;
    if (firstBlockIndex == lastBlockIndex) {
      break;
    }

    firstBlockIndex++;
    firstBlockOffset = 0;
  }
}

inline void load_first_normal(unit_t *scratch, const sets_t &table,
                              size_t gene) {
  const size_t numTumor = table.numTumor;
  const size_t numCols = table.numCols;

  const size_t rowOffset = gene * numCols;
  size_t startBit = rowOffset + numTumor;
  size_t endBit = rowOffset + (numCols - 1);

  size_t dstPos = 0;

  size_t firstBlockIndex = startBit / BITS_PER_UNIT;
  size_t firstBlockOffset = startBit % BITS_PER_UNIT;
  size_t lastBlockIndex = endBit / BITS_PER_UNIT;
  size_t lastBlockOffset = endBit % BITS_PER_UNIT;

  while (true) {
    size_t endOff = (firstBlockIndex == lastBlockIndex) ? lastBlockOffset
                                                        : (BITS_PER_UNIT - 1);

    unit_t blockVal = table.data[firstBlockIndex];

    if (firstBlockOffset != 0 || (firstBlockIndex == lastBlockIndex)) {
      unit_t rightMask;
      if (endOff >= (BITS_PER_UNIT - 1)) {
        rightMask = ALL_BITS_SET;
      } else {
        rightMask = (((unit_t)1 << (endOff + 1)) - (unit_t)1);
      }
      blockVal &= rightMask;
    }

    blockVal >>= firstBlockOffset;

    size_t usedBits = (endOff - firstBlockOffset + 1);

    size_t dstBlock = dstPos / BITS_PER_UNIT;
    size_t dstOff = dstPos % BITS_PER_UNIT;

    unit_t shiftedVal = blockVal << dstOff;

    size_t blockEnd = dstOff + usedBits - 1;
    if (blockEnd >= BITS_PER_UNIT) {
      blockEnd = BITS_PER_UNIT - 1;
    }
    unit_t localMask = 0ULL;
    if (dstOff <= blockEnd) {
      unit_t rm;
      if (blockEnd >= (BITS_PER_UNIT - 1)) {
        rm = ALL_BITS_SET;
      } else {
        rm = (((unit_t)1 << (blockEnd + 1)) - (unit_t)1);
      }
      unit_t lm;
      if (dstOff == 0) {
        lm = ALL_BITS_SET;
      } else {
        lm = ~(((unit_t)1 << dstOff) - (unit_t)1);
      }
      localMask = (rm & lm);
    }

    scratch[dstBlock] =
        (scratch[dstBlock] & ~localMask) | (shiftedVal & localMask);

    dstPos += usedBits;
    if (firstBlockIndex == lastBlockIndex) {
      break;
    }

    firstBlockIndex++;
    firstBlockOffset = 0;
  }
}**/

#endif
// ############MACROS####################
