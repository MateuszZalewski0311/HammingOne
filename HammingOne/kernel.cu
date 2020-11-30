// includes, system
#include <stdio.h>
#include <random>
#include <bitset>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <algorithm>

// includes, cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes, thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

////////////////////////////////////////////////////////////////////////////////
#define WORD_SIZE 23
#define DATA_SIZE 10000

////////////////////////////////////////////////////////////////////////////////
// function declarations
template<size_t N>
unsigned int hamming_distance(const typename std::bitset<N>& A, const typename std::bitset<N>& B);
template<size_t N>
typename std::bitset<N> random_bitset(double p);
template<size_t N, size_t M>
void generate_data(typename std::unordered_set<std::bitset<N>>& _data_uset, \
    const bool timeCount = true, const bool consoleOutput = true, const float p = 0.5f);
template<size_t N>
void find_ham1(const typename std::unordered_set<std::bitset<N>>& _data_uset, \
    typename std::vector<std::bitset<N>>& _ham1_pairs_1, typename std::vector<std::bitset<N>>& _ham1_pairs_2, \
    const bool timeCount = true, const bool pairsOutput = true, const bool consoleOutput = true);
//template<size_t N>
//void find_ham1_temp(const typename std::unordered_set<std::bitset<N>>& _data_uset, \
//    typename std::vector<std::bitset<N>>& _ham1_pairs_1, typename std::vector<std::bitset<N>>& _ham1_pairs_2, \
//    const bool timeCount = true, const bool pairsOutput = true, const bool consoleOutput = true);

////////////////////////////////////////////////////////////////////////////////
// word generating function
template<size_t N> // p = 0.5 gives equal chance for 0's and 1's to occur
typename std::bitset<N> random_bitset(double p) 
{
    typename std::bitset<N> bits;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(p);

    for (size_t i = 0; i < N; ++i) {
        bits[i] = dist(gen);
    }

    return bits;
}

////////////////////////////////////////////////////////////////////////////////
// data generating function
template<size_t N, size_t M>
void generate_data(typename std::unordered_set<std::bitset<N>>& _data_uset, \
    const bool timeCount, const bool consoleOutput, const float p)
{
    std::chrono::steady_clock::time_point start, finish;
    std::chrono::duration<double> elapsed;

    if (consoleOutput) std::cout << "Beginning Data Generation...\n";

    // Record start time
    if (consoleOutput && timeCount) start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < M; ++i)
    {
        while (false == (_data_uset.emplace(random_bitset<N>(p)).second));
    }

    // Record end time
    if (consoleOutput && timeCount) finish = std::chrono::high_resolution_clock::now();

    if (consoleOutput)
    {
        if (timeCount) elapsed = finish - start;
        std::cout << "Data Generation Finished!\n";
        if (timeCount) std::cout << "Elapsed time: " << elapsed.count() << " s\n";
        //std::cout << "Data has " << data.size() << " unique elements\n";
        std::cout << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
// data loading function
//
////////////////////////////////////////////////////////////////////////////////
// data saving function

////////////////////////////////////////////////////////////////////////////////
// finding pairs with hamming distance 1 function
template<size_t N>
void find_ham1(const typename std::unordered_set<std::bitset<N>>& _data_uset, \
    typename std::vector<std::bitset<N>>& _ham1_pairs_1, typename std::vector<std::bitset<N>>& _ham1_pairs_2, \
    const bool timeCount, const bool pairsOutput, const bool consoleOutput)
{
    std::chrono::steady_clock::time_point start, finish;
    std::chrono::duration<double> elapsed;

    if (consoleOutput) std::cout << "Looking for pairs with hamming distance 1 ...\n";

    // Record start time
    if (consoleOutput && timeCount) start = std::chrono::high_resolution_clock::now();

    unsigned int ham1 = 0;
    for (auto it1 = std::begin(_data_uset); it1 != std::end(_data_uset); ++it1)
    {
        for (auto it2 = std::next(it1); it2 != std::end(_data_uset); ++it2)
        {
            if (1 == hamming_distance<N>(*it1, *it2))
            {
                _ham1_pairs_1.emplace_back(std::bitset<N>(*it1));
                _ham1_pairs_2.emplace_back(std::bitset<N>(*it2));
                //_ham1_pairs_1.push_back(*it1);
                //_ham1_pairs_2.push_back(*it2);
                ++ham1;
            }
        }
    }

    // Record end time
    if (consoleOutput && timeCount) finish = std::chrono::high_resolution_clock::now();

    if (consoleOutput)
    {
        if (timeCount) elapsed = finish - start;
        std::cout << "Finished!\n";
        if (timeCount) std::cout << "Elapsed time: " << elapsed.count() << " s\n";
        std::cout << ham1 << " pairs found\n\n";
    }

    if (ham1 && pairsOutput && consoleOutput)
    {
        std::cout << "Pairs found:\n";

        for (auto it1 = std::begin(_ham1_pairs_1), it2 = std::begin(_ham1_pairs_2); it1 != std::end(_ham1_pairs_1); ++it1, ++it2)
        {
            std::cout << *it1 << " " << *it2 << std::endl;
        }

        std::cout << std::endl;
    }
}

//template<size_t N>
//void find_ham1_temp(const typename std::unordered_set<std::bitset<N>>& _data_uset, \
//    typename std::vector<std::bitset<N>>& _ham1_pairs_1, typename std::vector<std::bitset<N>>& _ham1_pairs_2, \
//    const bool timeCount, const bool pairsOutput, const bool consoleOutput)
//{
//    std::chrono::steady_clock::time_point start, finish;
//    std::chrono::duration<double> elapsed;
//
//    if (consoleOutput) std::cout << "Looking for pairs with hamming distance 1 ...\n";
//
//    // Record start time
//    if (consoleOutput && timeCount) start = std::chrono::high_resolution_clock::now();
//
//    unsigned int ham1 = 0;
//    for (const auto& A : _data_uset)
//    {
//        for (const auto& B : _data_uset)
//        {
//            if (1 == hamming_distance<N>(A, B))
//            {
//                auto it1 = std::find(std::begin(_ham1_pairs_2), std::end(_ham1_pairs_2), A);
//                auto it2 = std::find(std::begin(_ham1_pairs_1), std::end(_ham1_pairs_1), B);
//                if (it1 != std::end(_ham1_pairs_2) && it2 != std::end(_ham1_pairs_1) && it1 - std::begin(_ham1_pairs_2) == it2 - std::begin(_ham1_pairs_1)) {
//                    continue;
//                }
//                _ham1_pairs_1.emplace_back(std::bitset<N>(A));
//                _ham1_pairs_2.emplace_back(std::bitset<N>(B));
//                ++ham1;
//            }
//        }
//    }
//
//    // Record end time
//    if (consoleOutput && timeCount) finish = std::chrono::high_resolution_clock::now();
//
//    if (consoleOutput)
//    {
//        if (timeCount) elapsed = finish - start;
//        std::cout << "Finished!\n";
//        if (timeCount) std::cout << "Elapsed time: " << elapsed.count() << " s\n";
//        std::cout << ham1 << " pairs found\n\n";
//    }
//
//    if (ham1 && pairsOutput && consoleOutput)
//    {
//        std::cout << "Pairs found:\n";
//
//        for (auto it1 = std::begin(_ham1_pairs_1), it2 = std::begin(_ham1_pairs_2); it1 != std::end(_ham1_pairs_1); ++it1, ++it2)
//        {
//            std::cout << *it1 << " " << *it2 << std::endl;
//        }
//
//        std::cout << std::endl;
//    }
//}

////////////////////////////////////////////////////////////////////////////////
// hamming distance function
template<size_t N>
unsigned int hamming_distance(const typename std::bitset<N>& A, const typename std::bitset<N>& B)
{
    return (A ^ B).count();
}

////////////////////////////////////////////////////////////////////////////////
int main()
{
    //cudaError_t cudaStatus;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    /*cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }*/

    //thrust::host_vector<std::bitset<1024>> h_data(100000);
    std::unordered_set<std::bitset<WORD_SIZE>> data_uset;
    std::vector<std::bitset<WORD_SIZE>> ham1_pairs_1;
    std::vector<std::bitset<WORD_SIZE>> ham1_pairs_2;

    data_uset.reserve(DATA_SIZE);
    generate_data<WORD_SIZE, DATA_SIZE>(data_uset);

    find_ham1<WORD_SIZE>(data_uset, ham1_pairs_1, ham1_pairs_2, true, true);

    system("pause");

    return 0;
}

