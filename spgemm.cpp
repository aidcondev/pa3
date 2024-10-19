#include "functions.h"
#include <mpi.h>

#include <vector>
#include <utility>

void spgemm(int m, int h, int n, 
            std::vector<std::pair<std::pair<int,int>, int>> &A,  // A in COO format (already distributed)
            std::vector<std::pair<std::pair<int,int>, int>> &B,  // B in COO format (already distributed)
            std::vector<std::pair<std::pair<int,int>, int>> &C)  // Result C in COO format (expected to be distributed)
{ 
    // Implement your algorithm here

    return;
}