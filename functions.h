#include <iostream>
#include <vector>


void spgemm(int m, int h, int n, std::vector<std::pair<std::pair<int,int>, int>> &A, std::vector<std::pair<std::pair<int,int>, int>> &B, std::vector<std::pair<std::pair<int,int>, int>> &C);

void bfs(int nodes, int source, std::vector<std::pair<std::pair<int, int>, int>> &graph, std::vector<int> &result);

int triangle_counts(int nodes, std::vector<std::pair<std::pair<int, int>, int>> &graph_coo);