#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <limits.h>
#include <queue>
#include <mpi.h>
#include <unordered_set>
#include <map>

#include "functions.h"

bool equal(std::vector<std::pair<std::pair<int, int>, int>> &m1, std::vector<std::pair<std::pair<int, int>, int>> &m2)
{
    if (m1.size() != m2.size())
    {
        return 0;
    }
    for (int i = 0; i < m1.size(); i++)
    {
        if (m1[i].second != m2[i].second)
        {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string flag = argv[1];
    std::string filename = argv[2];

    int nodes, edges, source;
    std::vector<std::pair<int, int>> edge_list;
    std::vector<std::pair<std::pair<int, int>, int>> A;
    std::vector<std::pair<std::pair<int, int>, int>> A_T;
    int result_nnz;
    std::vector<std::pair<std::pair<int, int>, int>> spgemm_result;
    std::vector<int> bfs_result;
    int tc_result=0;

    

    if (rank == 0)
    {
        if (argc < 3)
        {
            std::cerr << "Usage: " << argv[0] << " <-flag> <input_file>" << std::endl;
            return 1;
        }

        if (flag != "-spgemm" && flag != "-bfs" && flag != "-tc")
        {
            std::cerr << "Invalid flag: " << flag << std::endl;
            return 1;
        }

        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Could not open file: " << filename << std::endl;
            return 1;
        }

        file >> nodes >> source >> edges;

        for (int i = 0; i < edges; ++i)
        {
            file >> edge_list.emplace_back().first >> edge_list.back().second;
        }

        for (int i = 0; i < edge_list.size(); i++)
        {
            A.push_back(std::make_pair(std::make_pair(edge_list[i].first, edge_list[i].second), 1));
            A_T.push_back(std::make_pair(std::make_pair(edge_list[i].second, edge_list[i].first), 1));
        }

        std::sort(A.begin(), A.end());
        std::sort(A_T.begin(), A_T.end());

        file >> result_nnz;

        for (int i = 0; i < result_nnz; ++i)
        {
            file >> spgemm_result.emplace_back().first.first >> spgemm_result.back().first.second >> spgemm_result.back().second;
        }


        bfs_result.resize(nodes);
        for (int i = 0; i < nodes; i++)
        {
            file >> bfs_result[i];
        }

        file >> tc_result;


        file.close();
    }

    MPI_Bcast(&nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (flag == "-spgemm")
    {
        int a_send_rows[size];
        int b_send_rows[size];
        int a_send_indices[size + 1];
        int b_send_indices[size + 1];
        int a_rows_to_rank[nodes];
        int b_rows_to_rank[nodes];
        int c_send_rows[size];
        int c_send_indices[size + 1];
        int c_rows_to_rank[nodes];

        std::vector<int> a_send_counts(size);
        std::vector<int> b_send_counts(size);
        std::vector<std::vector<int>> a_send_data(size);
        std::vector<std::vector<int>> b_send_data(size);
        std::vector<int> flat_a_send_data;
        std::vector<int> flat_b_send_data;
        std::vector<int> c_send_counts(size);
        std::vector<std::vector<int>> c_send_data(size);
        std::vector<int> flat_c_send_data;

        for (int i = 0; i < size; i++)
        {
            a_send_rows[i] = nodes / size;
            c_send_rows[i] = nodes / size;
            b_send_rows[i] = nodes / size;
            if (nodes % size > i)
            {
                a_send_rows[i]++;
                b_send_rows[i]++;
                c_send_rows[i]++;
            }
        }

        if (rank == 0)
        {
            a_send_indices[0] = 0;
            b_send_indices[0] = 0;
            c_send_indices[0] = 0;
            for (int i = 1; i < size + 1; i++)
            {
                a_send_indices[i] = a_send_indices[i - 1] + a_send_rows[i - 1];
                b_send_indices[i] = b_send_indices[i - 1] + b_send_rows[i - 1];
                c_send_indices[i] = c_send_indices[i - 1] + c_send_rows[i - 1];
            }

            for (int i = 0; i < size; i++)
            {
                for (int j = a_send_indices[i]; j < a_send_indices[i + 1]; j++)
                {
                    a_rows_to_rank[j] = i;
                }
                for (int j = b_send_indices[i]; j < b_send_indices[i + 1]; j++)
                {
                    b_rows_to_rank[j] = i;
                }
                for (int j = c_send_indices[i]; j < c_send_indices[i + 1]; j++)
                {
                    c_rows_to_rank[j] = i;
                }
            }

            for (int i = 0; i < A.size(); i++)
            {
                a_send_counts[a_rows_to_rank[A[i].first.first]] += 3;
                a_send_data[a_rows_to_rank[A[i].first.first]].push_back(A[i].first.first);
                a_send_data[a_rows_to_rank[A[i].first.first]].push_back(A[i].first.second);
                a_send_data[a_rows_to_rank[A[i].first.first]].push_back(A[i].second);
            }

            for (int i = 0; i < A_T.size(); i++)
            {
                b_send_counts[b_rows_to_rank[A_T[i].first.first]] += 3;
                b_send_data[b_rows_to_rank[A_T[i].first.first]].push_back(A_T[i].first.first);
                b_send_data[b_rows_to_rank[A_T[i].first.first]].push_back(A_T[i].first.second);
                b_send_data[b_rows_to_rank[A_T[i].first.first]].push_back(A_T[i].second);
            }

            for (int i = 0; i < spgemm_result.size(); i++)
            {
                c_send_counts[c_rows_to_rank[spgemm_result[i].first.first]] += 3;
                c_send_data[c_rows_to_rank[spgemm_result[i].first.first]].push_back(spgemm_result[i].first.first);
                c_send_data[c_rows_to_rank[spgemm_result[i].first.first]].push_back(spgemm_result[i].first.second);
                c_send_data[c_rows_to_rank[spgemm_result[i].first.first]].push_back(spgemm_result[i].second);
            }

            a_send_indices[0] = 0;
            b_send_indices[0] = 0;
            c_send_indices[0] = 0;
            for (int i = 1; i < size; i++)
            {
                a_send_indices[i] = a_send_indices[i - 1] + a_send_counts[i - 1];
                b_send_indices[i] = b_send_indices[i - 1] + b_send_counts[i - 1];
                c_send_indices[i] = c_send_indices[i - 1] + c_send_counts[i - 1];
            }

            for (const auto &vec : a_send_data)
            {
                flat_a_send_data.insert(flat_a_send_data.end(), vec.begin(), vec.end());
            }

            for (const auto &vec : b_send_data)
            {
                flat_b_send_data.insert(flat_b_send_data.end(), vec.begin(), vec.end());
            }

            for (const auto &vec : c_send_data)
            {
                flat_c_send_data.insert(flat_c_send_data.end(), vec.begin(), vec.end());
            }

        }

        int a_recv_count = 0;
        int b_recv_count = 0;
        int c_recv_count = 0;

        MPI_Scatter(a_send_counts.data(), 1, MPI_INT, &a_recv_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(b_send_counts.data(), 1, MPI_INT, &b_recv_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(c_send_counts.data(), 1, MPI_INT, &c_recv_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> a_recv_data(a_recv_count);
        std::vector<int> b_recv_data(b_recv_count);
        std::vector<int> c_recv_data(c_recv_count);

        MPI_Scatterv(&flat_a_send_data[0], a_send_counts.data(), a_send_indices, MPI_INT, &a_recv_data[0], a_recv_count, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(&flat_b_send_data[0], b_send_counts.data(), b_send_indices, MPI_INT, &b_recv_data[0], b_recv_count, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(&flat_c_send_data[0], c_send_counts.data(), c_send_indices, MPI_INT, &c_recv_data[0], c_recv_count, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<std::pair<std::pair<int, int>, int>> a_recv_data_coo;
        for (int i = 0; i < a_recv_data.size(); i += 3)
        {
            a_recv_data_coo.push_back(std::make_pair(std::make_pair(a_recv_data[i], a_recv_data[i + 1]), a_recv_data[i + 2]));
        }

        std::vector<std::pair<std::pair<int, int>, int>> b_recv_data_coo;
        for (int i = 0; i < b_recv_data.size(); i += 3)
        {
            b_recv_data_coo.push_back(std::make_pair(std::make_pair(b_recv_data[i], b_recv_data[i + 1]), b_recv_data[i + 2]));
        }

        std::vector<std::pair<std::pair<int, int>, int>> c_recv_data_coo;
        for (int i = 0; i < c_recv_data.size(); i += 3)
        {
            c_recv_data_coo.push_back(std::make_pair(std::make_pair(c_recv_data[i], c_recv_data[i + 1]), c_recv_data[i + 2]));
        }   

        std::vector<std::pair<std::pair<int, int>, int>> C_computed;
        double start = MPI_Wtime();

        spgemm(nodes, nodes, nodes, a_recv_data_coo, b_recv_data_coo, C_computed);

        double end = MPI_Wtime();
        double total_time = end - start;
        double total_time_max = 0;

        MPI_Scan(&total_time, &total_time_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (rank == size - 1){
            MPI_Send(&total_time_max, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        if (rank == 0){
            MPI_Recv(&total_time_max, 1, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        std::sort(c_recv_data_coo.begin(), c_recv_data_coo.end());
        std::sort(C_computed.begin(), C_computed.end());

        int flag = 0, total_flag = 0;
        if (c_recv_data_coo.size() == 0 && C_computed.size() == 0)
        {
            flag = 1;
        }
        else if (equal(c_recv_data_coo, C_computed))
        {
            flag = 1;
        }

        int C_send_count = C_computed.size();
        int total_C_computed_size = 0;

        MPI_Reduce(&C_send_count, &total_C_computed_size, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Reduce(&flag, &total_flag, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // Compare C vector and result vector for verification
        if (rank == 0)
        {
            if (total_flag == size)
            {
                std::cout << "SPGEMM is correct" << std::endl;
                std::cout << "Time taken by SPGEMM: " << total_time_max << std::endl;
            }
            else
            {
                std::cout << "SPGEMM is incorrect" << std::endl;
                std::cout << "Time taken by SPGEMM: " << total_time_max << std::endl;
            }
        }

        MPI_Finalize();
    }

    else if (flag == "-bfs")
    {

        std::vector<int> send_counts(size);
        int nodes_to_rank[nodes];
        int send_indices[size + 1];
        std::vector<std::vector<int>> send_data(size);
        std::vector<int> flat_send_data;

        int send_rows[size];
        for (int i = 0; i < size; i++)
        {
            send_rows[i] = nodes / size;
            if (nodes % size > i)
            {
                send_rows[i]++;
            }
        }

        if (rank == 0)
        {
            send_indices[0] = 0;
            for (int i = 1; i < size + 1; i++)
            {
                send_indices[i] = send_indices[i - 1] + send_rows[i - 1];
            }

            for (int i = 0; i < size; i++)
            {
                for (int j = send_indices[i]; j < send_indices[i + 1]; j++)
                {
                    nodes_to_rank[j] = i;
                }
            }

            for (int i = 0; i < edge_list.size(); i++)
            {
                send_counts[nodes_to_rank[edge_list[i].first]] += 3;
                send_data[nodes_to_rank[edge_list[i].first]].push_back(edge_list[i].first);
                send_data[nodes_to_rank[edge_list[i].first]].push_back(edge_list[i].second);
                send_data[nodes_to_rank[edge_list[i].first]].push_back(1);
            }

            send_indices[0] = 0;
            for (int i = 1; i < size; i++)
            {
                send_indices[i] = send_indices[i - 1] + send_counts[i - 1];
            }
            for (const auto &vec : send_data)
            {
                flat_send_data.insert(flat_send_data.end(), vec.begin(), vec.end());
            }
        }

        int recv_count = 0;
        MPI_Scatter(send_counts.data(), 1, MPI_INT, &recv_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> recv_data(recv_count);
        MPI_Scatterv(&flat_send_data[0], send_counts.data(), send_indices, MPI_INT, &recv_data[0], recv_count, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<std::pair<std::pair<int, int>, int>> recv_data_coo;
        for (int i = 0; i < recv_data.size(); i += 3)
        {
            recv_data_coo.push_back(std::make_pair(std::make_pair(recv_data[i], recv_data[i + 1]), recv_data[i + 2]));
        }

        std::vector<int> result_depth;

        int n_rows = nodes / size;

        if (nodes % size > rank)
        {
            n_rows++;
        }

        std::vector<int> node_depth(n_rows);
        std::vector<int> C_sendcounts(size);
        std::vector<int> displsC(size);

        int c_send_count = n_rows;
        
        MPI_Gather(&c_send_count, 1, MPI_INT, &C_sendcounts[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            displsC[0] = 0;
            for (int i = 1; i < size; ++i)
            {
                displsC[i] = displsC[i - 1] + C_sendcounts[i - 1];
            }
        }

        double start = MPI_Wtime();
        bfs(nodes, source, recv_data_coo, node_depth);
        double end = MPI_Wtime();
        double total_time = end - start;
        double total_time_max = 0;

        MPI_Scan(&total_time, &total_time_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (rank == size - 1){
            MPI_Send(&total_time_max, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        if (rank == 0){
            MPI_Recv(&total_time_max, 1, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }


        result_depth.resize(nodes);


        MPI_Gatherv(&node_depth[0], c_send_count, MPI_INT, &result_depth[0], &C_sendcounts[0], &displsC[0], MPI_INT, 0, MPI_COMM_WORLD);


        // Compare C vector and result vector for verification
        if (rank == 0)
        {
            if (std::equal(result_depth.begin(), result_depth.end(), bfs_result.begin()))
            {
                std::cout << "BFS is correct" << std::endl;
                std::cout << "Time taken by BFS: " << total_time_max << std::endl;
            }
            else
            {
                std::cout << "BFS is incorrect" << std::endl;
                std::cout << "Time taken by BFS: " << total_time_max << std::endl;
            }
        }

        MPI_Finalize();
    }

    else if (flag == "-tc")
    {
        std::vector<int> send_counts(size);
        int nodes_to_rank[nodes];
        int send_indices[size + 1];
        std::vector<std::vector<int>> send_data(size);
        std::vector<int> flat_send_data;

        int send_rows[size];
        for (int i = 0; i < size; i++)
        {
            send_rows[i] = nodes / size;
            if (nodes % size > i)
            {
                send_rows[i]++;
            }
        }

        if (rank == 0)
        {
            send_indices[0] = 0;
            for (int i = 1; i < size + 1; i++)
            {
                send_indices[i] = send_indices[i - 1] + send_rows[i - 1];
            }

            for (int i = 0; i < size; i++)
            {
                for (int j = send_indices[i]; j < send_indices[i + 1]; j++)
                {
                    nodes_to_rank[j] = i;
                }
            }

            for (int i = 0; i < edge_list.size(); i++)
            {
                send_counts[nodes_to_rank[edge_list[i].first]] += 3;
                send_data[nodes_to_rank[edge_list[i].first]].push_back(edge_list[i].first);
                send_data[nodes_to_rank[edge_list[i].first]].push_back(edge_list[i].second);
                send_data[nodes_to_rank[edge_list[i].first]].push_back(1);
            }

            send_indices[0] = 0;
            for (int i = 1; i < size; i++)
            {
                send_indices[i] = send_indices[i - 1] + send_counts[i - 1];
            }

            for (const auto &vec : send_data)
            {
                flat_send_data.insert(flat_send_data.end(), vec.begin(), vec.end());
            }

        }

        int recv_count = 0;
        MPI_Scatter(send_counts.data(), 1, MPI_INT, &recv_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> recv_data(recv_count);
        MPI_Scatterv(&flat_send_data[0], send_counts.data(), send_indices, MPI_INT, &recv_data[0], recv_count, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<std::pair<std::pair<int, int>, int>> recv_data_coo;
        for (int i = 0; i < recv_data.size(); i += 3)
        {
            recv_data_coo.push_back(std::make_pair(std::make_pair(recv_data[i], recv_data[i + 1]), recv_data[i + 2]));
        }

        int triangle_count = 0;
        double start = MPI_Wtime();
        triangle_count = triangle_counts(nodes, recv_data_coo);
        double end = MPI_Wtime();
        double total_time = end - start;
        double total_time_max = 0;

        MPI_Scan(&total_time, &total_time_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (rank == size - 1){
            MPI_Send(&total_time_max, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        if (rank == 0){
            MPI_Recv(&total_time_max, 1, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank == 0)
        {
            std::cout << "Triangle Count: " << triangle_count << std::endl;
            if (triangle_count == tc_result)
            {
                std::cout << "Triangle Counting is correct" << std::endl;
            }
            else
            {
                std::cout << "Triangle Counting is incorrect and not equal to " << tc_result << std::endl;
            }
            std::cout << "Time taken by Triangle Counting: " << total_time_max << std::endl;
        }

        MPI_Finalize();
    }
}