#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

typedef uint64_t unit_t;

struct sets_t {
  size_t numRows;
  size_t numTumor;
  size_t numNormal;
  size_t numCols;
  unit_t *tumorData;
  unit_t *normalData;
};

#define BITS_PER_UNIT (sizeof(unit_t) * CHAR_BIT)
#define ALL_BITS_SET (~(unit_t)0)
#define UNITS_FOR_BITS(N) (((N) + BITS_PER_UNIT - 1) / BITS_PER_UNIT)

inline bool is_empty(unit_t *buf, size_t validBits) {
  size_t fullUnits = validBits / BITS_PER_UNIT;
  size_t remainder = validBits % BITS_PER_UNIT;
  for (size_t i = 0; i < fullUnits; i++) {
    if (buf[i] != (unit_t)0)
      return false;
  }
  if (remainder > 0) {
    unit_t mask = ((unit_t)1 << remainder) - (unit_t)1;
    if ((buf[fullUnits] & mask) != 0)
      return false;
  }
  return true;
}

inline int bitCollection_size(unit_t *buf, size_t validBits) {
  size_t fullUnits = validBits / BITS_PER_UNIT;
  size_t remainder = validBits % BITS_PER_UNIT;
  int count = 0;
  for (size_t i = 0; i < fullUnits; i++) {
    count += __builtin_popcountll(buf[i]);
  }
  if (remainder > 0) {
    unit_t mask = ((unit_t)1 << remainder) - (unit_t)1;
    count += __builtin_popcountll(buf[fullUnits] & mask);
  }
  return count;
}

inline void load_first_tumor(unit_t *scratch, sets_t &table, size_t gene) {
  size_t rowUnits = UNITS_FOR_BITS(table.numTumor);
  size_t baseIdx = gene * rowUnits;
  std::memcpy(scratch, &table.tumorData[baseIdx], rowUnits * sizeof(unit_t));
}

inline void load_first_normal(unit_t *scratch, sets_t &table, size_t gene) {
  size_t rowUnits = UNITS_FOR_BITS(table.numNormal);
  size_t baseIdx = gene * rowUnits;
  std::memcpy(scratch, &table.normalData[baseIdx], rowUnits * sizeof(unit_t));
}

inline void inplace_intersect_tumor(unit_t *scratch, sets_t &table,
                                    size_t gene) {
  size_t rowUnits = UNITS_FOR_BITS(table.numTumor);
  size_t baseIdx = gene * rowUnits;
  for (size_t b = 0; b < rowUnits; b++) {
    scratch[b] &= table.tumorData[baseIdx + b];
  }
}

inline void inplace_intersect_normal(unit_t *scratch, sets_t &table,
                                     size_t gene) {
  size_t rowUnits = UNITS_FOR_BITS(table.numNormal);
  size_t baseIdx = gene * rowUnits;
  for (size_t b = 0; b < rowUnits; b++) {
    scratch[b] &= table.normalData[baseIdx + b];
  }
}

void print_units(unit_t *buf, size_t num_units, size_t validBits) {
  std::cout << "Data (in hex): ";
  for (size_t i = 0; i < num_units; i++) {
    std::cout << std::hex << buf[i] << " ";
  }
  std::cout << std::dec << "\n";
  std::cout << "Total set bits (over " << validBits
            << " bits): " << bitCollection_size(buf, validBits) << "\n";
}

int main() {
  sets_t table;
  table.numTumor = 70;
  table.numNormal = 50;
  table.numCols = 2;
  table.numRows = 2;
  size_t tumor_units_per_gene = UNITS_FOR_BITS(table.numTumor);
  size_t normal_units_per_gene = UNITS_FOR_BITS(table.numNormal);
  size_t max_units = tumor_units_per_gene > normal_units_per_gene
                         ? tumor_units_per_gene
                         : normal_units_per_gene;
  table.tumorData = new unit_t[table.numCols * tumor_units_per_gene];
  table.normalData = new unit_t[table.numCols * normal_units_per_gene];
  table.tumorData[0] = 0xAAAAAAAAAAAAAAAAULL;
  table.tumorData[1] = 0x3FULL;
  table.tumorData[2] = 0x0F0F0F0F0F0F0F0FULL;
  table.tumorData[3] = 0x15ULL;
  table.normalData[0] = 0xAAAAAAAAAAAAAAAAULL;
  table.normalData[1] = 0x0F0F0F0F0F0F0F0FULL;
  {
    unit_t test_buf[2] = {0, 0};
    std::cout << "is_empty on zeroed buffer (expect true): " << std::boolalpha
              << is_empty(test_buf, 70) << "\n";
    test_buf[0] = 1;
    std::cout << "is_empty after setting a bit (expect false): "
              << std::boolalpha << is_empty(test_buf, 70) << "\n";
  }
  {
    unit_t test_buf[2] = {0xF0F0F0F0F0F0F0F0ULL, 0x3FULL};
    int count = bitCollection_size(test_buf, 70);
    std::cout << "bitCollection_size for test pattern: " << count << "\n";
  }
  {
    unit_t *scratch = new unit_t[max_units];
    load_first_tumor(scratch, table, 0);
    std::cout << "\nTumor gene 0 loaded:\n";
    print_units(scratch, tumor_units_per_gene, table.numTumor);
    inplace_intersect_tumor(scratch, table, 1);
    std::cout << "After intersecting with tumor gene 1:\n";
    print_units(scratch, tumor_units_per_gene, table.numTumor);
    load_first_normal(scratch, table, 0);
    std::cout << "\nNormal gene 0 loaded:\n";
    print_units(scratch, normal_units_per_gene, table.numNormal);
    inplace_intersect_normal(scratch, table, 1);
    std::cout << "After intersecting with normal gene 1:\n";
    print_units(scratch, normal_units_per_gene, table.numNormal);
    delete[] scratch;
  }
  delete[] table.tumorData;
  delete[] table.normalData;
  return 0;
}
