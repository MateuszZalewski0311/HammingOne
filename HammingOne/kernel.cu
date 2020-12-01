// includes, system
#include <stdio.h>
#include <random>
#include <bitset>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <algorithm>
#include <limits>

// includes, cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes, thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

////////////////////////////////////////////////////////////////////////////////
#define WORD_SIZE 10
#define DATA_SIZE 50
#define UINT_BITSIZE (unsigned int)(8*sizeof(unsigned int))
#define SUBWORDS_PER_WORD(N) (unsigned int)(std::ceil((float)N / (sizeof(unsigned int) * 8.0f)))

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
    const bool timeCount = true, const bool pairsOutput = true);
template<size_t N, size_t M>
thrust::device_vector<unsigned int> move_data_to_GPU(const typename std::unordered_set<std::bitset<N>>& data_uset);
__global__ void find_ham1_GPU_ker(const unsigned int* subwords, unsigned int* pair_flags);
template<size_t N>
void find_ham1_GPU(thrust::device_vector<unsigned int>& d_subwords, \
    thrust::device_vector<unsigned int>& d_pair_flags, \
    thrust::host_vector<unsigned int>& h_pair_flags, size_t pair_flags_size, \
    const bool timeCount, const bool pairsOutput, const typename std::unordered_set<std::bitset<N>>& _data_uset);
template<size_t N>
void print_pairs_from_flags(thrust::host_vector<unsigned int>& h_pair_flags, size_t pair_flags_size, \
    const typename std::unordered_set<std::bitset<N>>& _data_uset);

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
        std::cout << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
// data loading function
//
////////////////////////////////////////////////////////////////////////////////
// data saving function

////////////////////////////////////////////////////////////////////////////////
// finding pairs with hamming distance 1 on CPU
template<size_t N>
void find_ham1(const typename std::unordered_set<std::bitset<N>>& _data_uset, \
    typename std::vector<std::bitset<N>>& _ham1_pairs_1, typename std::vector<std::bitset<N>>& _ham1_pairs_2, \
    const bool timeCount, const bool pairsOutput)
{
    std::chrono::steady_clock::time_point start, finish;
    std::chrono::duration<double> elapsed;

    std::cout << "Looking for pairs with hamming distance 1 ...\n";

    // Record start time
    if (timeCount) start = std::chrono::high_resolution_clock::now();

    unsigned int ham1 = 0;
    for (auto it1 = std::begin(_data_uset); it1 != std::end(_data_uset); ++it1)
    {
        for (auto it2 = std::next(it1); it2 != std::end(_data_uset); ++it2)
        {
            if (1 == hamming_distance<N>(*it1, *it2))
            {
                _ham1_pairs_1.emplace_back(std::bitset<N>(*it1));
                _ham1_pairs_2.emplace_back(std::bitset<N>(*it2));
                ++ham1;
            }
        }
    }

    // Record end time
    if (timeCount) finish = std::chrono::high_resolution_clock::now();
    if (timeCount) elapsed = finish - start;

    std::cout << "Finished!\n";
    if (timeCount) std::cout << "Elapsed time: " << elapsed.count() << " s\n";
    std::cout << ham1 << " pairs found\n\n";

    if (ham1 && pairsOutput)
    {
        std::cout << "Pairs found:\n";

        for (auto it1 = std::begin(_ham1_pairs_1), it2 = std::begin(_ham1_pairs_2); it1 != std::end(_ham1_pairs_1); ++it1, ++it2)
        {
            std::cout << *it1 << " " << *it2 << std::endl;
        }

        std::cout << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
// hamming distance function
template<size_t N>
unsigned int hamming_distance(const typename std::bitset<N>& A, const typename std::bitset<N>& B)
{
    return (A ^ B).count();
}

////////////////////////////////////////////////////////////////////////////////
// move data to gpu
template<size_t N, size_t M>
thrust::device_vector<unsigned int> move_data_to_GPU(const typename std::unordered_set<std::bitset<N>>& data_uset)
{
    //N - WORD_SIZE, M - DATA_SIZE
    thrust::host_vector<unsigned int> h_words(M * SUBWORDS_PER_WORD(N));
    thrust::device_vector<unsigned int> d_words;

    std::chrono::steady_clock::time_point start, finish;
    std::chrono::duration<double> elapsed;

    // Record start time
    start = std::chrono::high_resolution_clock::now();

    int i = 0;
    for (const auto& word_bitset : data_uset)
    {
        //std::cout << std::endl << "Original " << word_bitset.to_string() << std::endl;
        if (N < UINT_BITSIZE)
        {
            std::string subword_str = word_bitset.to_string().substr(0, N);
            for (size_t subword_str_size = N; subword_str_size < UINT_BITSIZE; ++subword_str_size)
                subword_str += "0";
            unsigned int subword = (unsigned int)(std::bitset<N>(subword_str).to_ulong());
            //std::cout << "Subword: " << subword_str << ", " << subword << std::endl;
            h_words[i++] = subword;
            continue;
        }
        size_t j = 0;
        for (; j + UINT_BITSIZE < N; j += UINT_BITSIZE)
        {
            std::string subword_str = word_bitset.to_string().substr(j, UINT_BITSIZE);
            unsigned int subword = (unsigned int)(std::bitset<N>(subword_str).to_ulong());
            //std::cout << "Subword: " << subword_str << ", " << subword << std::endl;
            h_words[i++] = subword;
        }
        if (j + UINT_BITSIZE != N) // last subword smaller than UINT_BITSIZE
        {
            std::string subword_str = word_bitset.to_string().substr(j, N - j);
            for (size_t subword_str_size = N - j; subword_str_size < UINT_BITSIZE; ++subword_str_size)
                subword_str += "0";
            unsigned int subword = (unsigned int)(std::bitset<N>(subword_str).to_ulong());
            //std::cout << "Subword: " << subword_str << ", " << subword << std::endl;
            h_words[i++] = subword;
        }
    }
    d_words = h_words;

    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;

    std::cout << std::endl << "Data moved to GPU" << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl << std::endl;
    return d_words;
}

////////////////////////////////////////////////////////////////////////////////
// HammingOne kernel
__global__ void find_ham1_GPU_ker(const unsigned int* subwords, unsigned int* pair_flags)
{
    const unsigned int word_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int subwords_per_word = SUBWORDS_PER_WORD(WORD_SIZE);

    if (word_idx >= DATA_SIZE)
        return;

    unsigned int* word = new unsigned int[subwords_per_word];
    for (size_t i = 0; i < subwords_per_word; ++i)
    {
        word[i] = subwords[word_idx * subwords_per_word + i];
    }

    unsigned int hamming_distance, flag_subword_offset, flag_in_subword;

    for (size_t comparison_idx = word_idx + 1; comparison_idx < DATA_SIZE; ++comparison_idx)
    {
        hamming_distance = 0;
        for (size_t i = 0; i < subwords_per_word && hamming_distance < 2; ++i)
        {
            hamming_distance += __popc(word[i] ^ subwords[comparison_idx * subwords_per_word + i]);
        }
        if (!(hamming_distance >> 1)) // true when hamming_distance == 1
        {
            flag_subword_offset = comparison_idx / UINT_BITSIZE;
            flag_in_subword = 1 << UINT_BITSIZE - 1 - comparison_idx % UINT_BITSIZE;
            pair_flags[word_idx * subwords_per_word + flag_subword_offset] |= flag_in_subword;
        }
    }

    delete[] word;
}

////////////////////////////////////////////////////////////////////////////////
// finding pairs with hamming distance 1 on GPU
template<size_t N>
void find_ham1_GPU(thrust::device_vector<unsigned int>& d_subwords, \
    thrust::device_vector<unsigned int>& d_pair_flags, \
    thrust::host_vector<unsigned int>& h_pair_flags, size_t pair_flags_size, \
    const bool timeCount, const bool pairsOutput, const typename std::unordered_set<std::bitset<N>>& _data_uset)
{
    unsigned int threads = 512;
    unsigned int blocks = (unsigned int)std::ceil(DATA_SIZE / (double)threads);
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    unsigned int pairs_count = 0;
    auto d_subwords_ptr = thrust::raw_pointer_cast(d_subwords.begin().base());
    auto d_pair_flags_ptr = thrust::raw_pointer_cast(d_pair_flags.begin().base());
    float elapsed;
    cudaEvent_t start, stop;

    if (timeCount) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    std::cout << "Looking for pairs with hamming distance 1 ...\n";

    if (timeCount) cudaEventRecord(start, 0);
    find_ham1_GPU_ker<<<dimGrid, dimBlock>>>(d_subwords_ptr, d_pair_flags_ptr);
    if (timeCount) cudaEventRecord(stop, 0);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();

    if (timeCount) cudaEventElapsedTime(&elapsed, start, stop);

    std::cout << "Finished!\n";
    if (timeCount) std::cout << "Elapsed time: " << elapsed << " ms\n";

    h_pair_flags = d_pair_flags;
    for (size_t i = 0; i < pair_flags_size; ++i)
    {
        pairs_count += __popcnt(h_pair_flags[i]);
    }

    std::cout << pairs_count << " pairs found\n\n";

    if (pairs_count && pairsOutput)
        print_pairs_from_flags<N>(h_pair_flags, pair_flags_size, _data_uset);

    if (timeCount) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    thrust::fill(d_pair_flags.begin(), d_pair_flags.end(), 0);
}

////////////////////////////////////////////////////////////////////////////////
// pairs_flag to pairs output
template<size_t N>
void print_pairs_from_flags(thrust::host_vector<unsigned int>& h_pair_flags, size_t pair_flags_size, \
    const typename std::unordered_set<std::bitset<N>>& _data_uset)
{
    unsigned int subwords_per_word_flags = (unsigned int)std::ceil((double)DATA_SIZE / (double)UINT_BITSIZE);

    std::cout << "Pairs found:\n";

    for (size_t word_idx = 0; word_idx < DATA_SIZE; ++word_idx)
    {
        bool flag_found = false;
        unsigned int* word_flags = new unsigned int[subwords_per_word_flags];
        for (size_t i = 0; i < subwords_per_word_flags; ++i)
        {
            word_flags[i] = h_pair_flags[word_idx * subwords_per_word_flags + i];
        }
        for (size_t i = 0; i < subwords_per_word_flags; ++i)
            if (word_flags[i]) {
                flag_found = true;
                break;
            }
        if (!flag_found) continue;
        for (int i = subwords_per_word_flags-1; i >= 0; --i)
        {
            if (!word_flags[i])
                continue;
            int flags_set = __popcnt(h_pair_flags[i]);
            int flag_pos = DATA_SIZE - 2;
            for (size_t j = 0; j < flags_set;)
            {
                if (word_flags[i] % 2) {
                    std::cout << *std::next(std::begin(_data_uset), word_idx) << " " << *std::next(std::begin(_data_uset), flag_pos) << std::endl;
                    ++j;
                }
                word_flags[i] = word_flags[i] >> 1;
                --flag_pos;
            }
        }
        delete[] word_flags;
    }

    std::cout << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
int main()
{
    bool updated_data_GPU = true;
    unsigned short menu_choice = 0;
    size_t pair_flags_size = DATA_SIZE * (std::ceil((double)DATA_SIZE / (double)UINT_BITSIZE));
    thrust::device_vector<unsigned int> d_subwords;
    thrust::device_vector<unsigned int> d_pair_flags(pair_flags_size, 0);
    thrust::host_vector<unsigned int> h_pair_flags;
    std::unordered_set<std::bitset<WORD_SIZE>> data_uset;
    std::vector<std::bitset<WORD_SIZE>> ham1_pairs_1;
    std::vector<std::bitset<WORD_SIZE>> ham1_pairs_2;

    while (menu_choice != 5) {
        std::cout << "1. Generate Data" << std::endl;
        std::cout << "2. Save/Load Data" << std::endl;
        if (!data_uset.empty()) {
            if (d_subwords.empty())
                std::cout << "3. Move Data to GPU - !!! No Data on GPU !!!" << std::endl;
            else if (!updated_data_GPU)
                std::cout << "3. Move Data to GPU - !!! Data on GPU not matching Data on CPU !!!" << std::endl;
            else
                std::cout << "3. Move Data to GPU" << std::endl;
            std::cout << "4. Find Pairs" << std::endl;
        }
        else {
            std::cout << "3. Move Data to GPU - !!! Generate/Load Data before attempting to move the data to GPU !!!" << std::endl;
            std::cout << "4. Find Pairs - !!! Generate/Load Data before attempting to find pairs !!!" << std::endl;
        }
        std::cout << "5. Exit" << std::endl;
        std::cout << "6. Clear Console" << std::endl;
        std::cout << "Choice: ";
        std::cin >> menu_choice;
        switch (menu_choice)
        {
        case 1:
            if (!data_uset.empty())
                data_uset.clear();
            data_uset.reserve(DATA_SIZE);
            std::cout << std::endl;
            generate_data<WORD_SIZE, DATA_SIZE>(data_uset);
            updated_data_GPU = false;
            break;
        case 2:
            std::cout << std::endl << "Not implemented yet :(" << std::endl << std::endl;
            break;
        case 3:
            if (!data_uset.empty()) {
                d_subwords = move_data_to_GPU<WORD_SIZE, DATA_SIZE>(data_uset);
                updated_data_GPU = true;
            }
            else
                std::cout << std::endl << "!!! Generate / Load Data before attempting to move the data to GPU !!!" << std::endl << std::endl;
            break;
        case 4:
            std::cout << std::endl;
            if (!data_uset.empty()) {
                while (menu_choice != 3)
                {
                    std::cout << "1. Use CPU" << std::endl;
                    if (d_subwords.empty())
                        std::cout << "2. Use GPU - !!! No Data on GPU !!!" << std::endl;
                    else if (!updated_data_GPU)
                        std::cout << "2. Use GPU - !!! Data on GPU not matching Data on CPU !!!" << std::endl;
                    else
                        std::cout << "2. Use GPU" << std::endl;
                    std::cout << "3. Go Back" << std::endl;
                    std::cout << "Choice: ";
                    std::cin >> menu_choice;
                    std::cout << std::endl;
                    switch (menu_choice)
                    {
                    case 1:
                        char c;
                        do {
                            std::cout << "Output pairs to console? (y/n):";
                            std::cin.clear();
                            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                            c = std::getc(stdin);
                            if (c == 'y' || c == 'Y') {
                                find_ham1<WORD_SIZE>(data_uset, ham1_pairs_1, ham1_pairs_2, true, true);
                                break;
                            }
                            else if (c == 'n' || c == 'N') {
                                find_ham1<WORD_SIZE>(data_uset, ham1_pairs_1, ham1_pairs_2, true, false);
                                break;
                            }
                            std::cout << "Please provide a valid choice" << std::endl;
                        } while (true);
                        break;
                    case 2:
                        if (d_subwords.empty())
                            std::cout << std::endl << "!!! No Data on GPU !!!" << std::endl << std::endl;
                        else do {
                            std::cout << "Output pairs to console? (y/n):";
                            std::cin.clear();
                            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                            c = std::getc(stdin);
                            if (c == 'y' || c == 'Y') {
                                find_ham1_GPU<WORD_SIZE>(d_subwords, d_pair_flags, h_pair_flags, pair_flags_size, true, true, data_uset);
                                break;
                            }
                            else if (c == 'n' || c == 'N') {
                                find_ham1_GPU<WORD_SIZE>(d_subwords, d_pair_flags, h_pair_flags, pair_flags_size, true, false, data_uset);
                                break;
                            }
                            std::cout << "Please provide a valid choice" << std::endl;
                        } while (true);
                        break;
                    case 3:
                        break;
                    default:
                        std::cout << "Please provide a valid choice" << std::endl << std::endl;
                        break;
                    }
                }
            }
            else
                std::cout << std::endl << "!!! Generate/Load Data before attempting to find pairs !!!" << std::endl << std::endl;
            menu_choice = 4;
            break;
        case 5:
            break;
        case 6:
            system("CLS");
            break;
        default:
            std::cout << std::endl << "Please provide a valid choice" << std::endl << std::endl;
            break;
        }
    }

    return 0;
}