// includes, system
#include <stdio.h>
#include <random>
#include <bitset>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <algorithm>
#include <limits>
#include <fstream>

// includes, cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes, thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

////////////////////////////////////////////////////////////////////////////////
#define WORD_SIZE 1000
#define DATA_SIZE 100000
#define UINT_BITSIZE (unsigned int)(8*sizeof(unsigned int))
#define SUBWORDS_PER_WORD(N) (unsigned int)(std::ceil((float)N / (sizeof(unsigned int) * 8.0f)))

////////////////////////////////////////////////////////////////////////////////
// function declarations
template<size_t N>
unsigned int hamming_distance(const typename std::bitset<N>& A, const typename std::bitset<N>& B);
template<size_t N>
typename std::bitset<N> random_bitset(double p);
template<size_t N, size_t M>
void generate_random_data(typename std::vector<std::bitset<N>>& _data_vec, \
    const bool timeCount = true, const bool consoleOutput = true, const float p = 0.5f);
template<size_t N, size_t M>
void generate_predictable_data(typename std::vector<std::bitset<N>>& _data_vec, \
    const bool timeCount = true, const bool consoleOutput = true);
template<size_t N, size_t M>
void load_data(const char* words_filepath, const char* pairs_filepath, typename std::vector<std::bitset<N>>& _data_vec, \
    typename std::vector<std::pair<std::bitset<N>, std::bitset<N>>>& ham1_pairs);
template<size_t N, size_t M>
void save_data(const char* words_filepath, const char* pairs_filepath, const typename std::vector<std::bitset<N>>& _data_vec, \
    const typename std::vector<std::pair<std::bitset<N>, std::bitset<N>>>& ham1_pairs);
template<size_t N>
void find_ham1(const typename std::vector<std::bitset<N>>& _data_vec, \
    typename std::vector<std::pair<std::bitset<N>, std::bitset<N>>>& ham1_pairs, \
    const bool timeCount = true, const bool pairsOutput = true);
template<size_t N, size_t M>
thrust::device_vector<unsigned int> move_data_to_GPU(const typename std::vector<std::bitset<N>>& data_vec);
__global__ void find_ham1_GPU_ker(const unsigned int* subwords, unsigned int* pair_flags, const unsigned int subwords_per_pair_flags);
__global__ void count_ones(unsigned int* d_data, size_t pair_flags_size);
template<size_t N>
void find_ham1_GPU(thrust::device_vector<unsigned int>& d_subwords, \
    thrust::device_vector<unsigned int>& d_pair_flags, \
    thrust::host_vector<unsigned int>& h_pair_flags, size_t pair_flags_size, \
    const bool timeCount, const bool pairsOutput, const typename std::vector<std::bitset<N>>& _data_vec);
template<size_t N>
void process_pairs_from_flags(thrust::host_vector<unsigned int>& h_pair_flags, size_t pair_flags_size, \
    const typename std::vector<std::bitset<N>>& data_vec);

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
// fins MSB index in bitset
template<size_t N>
unsigned int MSB(std::bitset<N> bitset)
{
    if (!bitset.to_ulong())
        return 0;

    int msb = 0;
    while (bitset.to_ulong() != 1)
    {
        bitset >>= 1;
        ++msb;
    }
    return msb;
}

////////////////////////////////////////////////////////////////////////////////
// data generating function
template<size_t N, size_t M>
void generate_random_data(typename std::vector<std::bitset<N>>& _data_vec, \
    const bool timeCount, const bool consoleOutput, const float p)
{
    std::unordered_set<std::bitset<N>> data_uset;
    data_uset.reserve(M);

    std::chrono::steady_clock::time_point start, finish;
    std::chrono::duration<double> elapsed;

    if (consoleOutput) std::cout << "Beginning Data Generation...\n";

    // Record start time
    if (consoleOutput && timeCount) start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < M; ++i)
    {
        while (false == (data_uset.emplace(random_bitset<N>(p)).second));
    }

    // Record end time
    if (consoleOutput && timeCount) finish = std::chrono::high_resolution_clock::now();

    // Copy to vector
    for (const auto& it : data_uset) 
    {
        _data_vec.emplace_back(it);
    }

    if (consoleOutput)
    {
        if (timeCount) elapsed = finish - start;
        std::cout << "Data Generation Finished!\n";
        if (timeCount) std::cout << "Elapsed time: " << elapsed.count() << " s\n";
        std::cout << std::endl;
    }
}

template<size_t N, size_t M>
void generate_predictable_data(typename std::vector<std::bitset<N>>& _data_vec, \
    const bool timeCount, const bool consoleOutput)
{
    std::unordered_set<std::bitset<N>> data_uset;
    data_uset.reserve(M);

    std::chrono::steady_clock::time_point start, finish;
    std::chrono::duration<double> elapsed;

    unsigned int starting_word_counter = 0;
    std::bitset<N> starting_word;
    std::bitset<N> current_word;

    if (consoleOutput) std::cout << "Beginning Data Generation...\n";

    // Record start time
    if (consoleOutput && timeCount) start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < M;)
    {
        starting_word = std::bitset<N>(starting_word_counter++);
        if (data_uset.emplace(starting_word).second) {
            _data_vec.emplace_back(starting_word);
            ++i;
        }
        for (size_t j = starting_word.to_ulong() == 0 ? 0 : MSB(starting_word) + 1; j < N && i < M; ++j)
        {
            current_word = std::bitset<N>(0);
            current_word[j] = 1;
            current_word = current_word ^ starting_word;
            if (data_uset.emplace(current_word).second) {
                _data_vec.emplace_back(current_word);
                ++i;
            }
        }
    }

    // Record end time
    if (consoleOutput && timeCount) finish = std::chrono::high_resolution_clock::now();

    if (consoleOutput)
    {
        if (timeCount) elapsed = finish - start;
        std::cout << "Data Generation Finished!\n";
        if (_data_vec.size() != M) std::cout << "Wrong Data Size: " << _data_vec.size() << std::endl;
        if (timeCount) std::cout << "Elapsed time: " << elapsed.count() << " s\n";
        std::cout << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
// data loading function
template<size_t N, size_t M>
void load_data(const char* words_filepath, const char* pairs_filepath, typename std::vector<std::bitset<N>>& _data_vec, \
    typename std::vector<std::pair<std::bitset<N>, std::bitset<N>>>& ham1_pairs)
{
    size_t pairs_count = 0;
    std::string line, number;
    std::string separator = ";";
    size_t sep_pos = 0;
    std::ifstream words_file;
    words_file.open(words_filepath);
    if (!words_file.good()) {
        std::cout << "Error opening words_file\n\n";
        return;
    }
    _data_vec.clear();
    std::getline(words_file, line);
    std::getline(words_file, line);
    sep_pos = line.find(separator);
    if (sep_pos == std::string::npos) {
        std::cout << "Error(words_file) - wrong formatting\n\n";
        return;
    }
    if (std::stoi(line.substr(0, sep_pos)) != N) {
        std::cout << "Error(words_file) - WORD_SIZE different\n\n";
        return;
    }
    if (std::stoi(line.substr(sep_pos + 1)) != M) {
        std::cout << "Error(words_file) - DATA_SIZE different\n\n";
        return;
    }
    for (size_t i = 0; i < M; ++i) {
        std::getline(words_file, line);
        _data_vec.emplace_back(std::bitset<N>(line));
    }

    std::ifstream pairs_file;
    pairs_file.open(pairs_filepath);
    if (!words_file.good()) {
        std::cout << "Error opening pairs_file\n\n";
        return;
    }
    ham1_pairs.clear();
    std::getline(pairs_file, line);
    std::getline(pairs_file, line);
    sep_pos = line.find(separator);
    if (sep_pos == std::string::npos) {
        std::cout << "Error(pairs_file) - wrong formatting\n\n";
        return;
    }
    if (std::stoi(line.substr(0, sep_pos)) != N) {
        std::cout << "Error(pairs_file) - WORD_SIZE different\n\n";
        return;
    }
    pairs_count = std::stoi(line.substr(sep_pos + 1));
    for (size_t i = 0; i < pairs_count; ++i) {
        std::getline(pairs_file, line);
        sep_pos = line.find(separator);
        ham1_pairs.emplace_back(std::make_pair<std::bitset<N>, std::bitset<N>>(std::bitset<N>(line.substr(0, sep_pos)), std::bitset<N>(line.substr(sep_pos + 1))));
    }
    pairs_file.close();

    std::cout << "Loading Data successful!" << std::endl << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
// data saving function
template<size_t N, size_t M>
void save_data(const char* words_filepath, const char* pairs_filepath, const typename std::vector<std::bitset<N>>& _data_vec, \
    const typename std::vector<std::pair<std::bitset<N>, std::bitset<N>>>& ham1_pairs)
{
    if (_data_vec.empty()) {
        std::cout << "Words vector is empty";
    }
    std::ofstream words_file;
    std::remove(words_filepath);
    words_file.open(words_filepath);
    words_file << "WORD_SIZE;DATA_SIZE\n";
    words_file << N << ';' << M << "\n";
    for (size_t i = 0; i < M; ++i)
        words_file << _data_vec[i].to_string() << "\n";
    words_file.close();

    if (ham1_pairs.empty()) {
        std::cout << "Saving Data successful!" << std::endl << std::endl;
        return;
    }
    std::ofstream pairs_file;
    std::remove(pairs_filepath);
    pairs_file.open(pairs_filepath);
    pairs_file << "WORD_SIZE;PAIRS_COUNT\n";
    pairs_file << N << ';' << ham1_pairs.size() << "\n";
    for (size_t i = 0; i < ham1_pairs.size(); ++i)
        pairs_file << ham1_pairs[i].first.to_string() << ';' << ham1_pairs[i].second.to_string() << "\n";
    pairs_file.close();

    std::cout << "Saving Data successful!" << std::endl << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
// finding pairs with hamming distance 1 on CPU
template<size_t N>
void find_ham1(const typename std::vector<std::bitset<N>>& data_vec, \
    typename std::vector<std::pair<std::bitset<N>, std::bitset<N>>>& ham1_pairs, \
    const bool timeCount, const bool pairsOutput)
{
    std::chrono::steady_clock::time_point start, finish;
    std::chrono::duration<double> elapsed;

    std::cout << "Looking for pairs with hamming distance 1 ...\n";

    ham1_pairs.clear();

    // Record start time
    if (timeCount) start = std::chrono::high_resolution_clock::now();

    unsigned int ham1 = 0;
    for (auto it1 = std::begin(data_vec); it1 != std::end(data_vec); ++it1)
    {
        for (auto it2 = std::next(it1); it2 != std::end(data_vec); ++it2)
        {
            if (1 == hamming_distance<N>(*it1, *it2))
            {
                ham1_pairs.emplace_back(std::make_pair<std::bitset<N>, std::bitset<N>>(std::bitset<N>(*it1), std::bitset<N>(*it2)));
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

        for (const auto& it : ham1_pairs)
        {
            std::cout << it.first << " " << it.second << std::endl;
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
thrust::device_vector<unsigned int> move_data_to_GPU(const typename std::vector<std::bitset<N>>& data_vec)
{
    //N - WORD_SIZE, M - DATA_SIZE
    thrust::host_vector<unsigned int> h_words(M * SUBWORDS_PER_WORD(N));
    thrust::device_vector<unsigned int> d_words;

    std::chrono::steady_clock::time_point start, finish;
    std::chrono::duration<double> elapsed;

    // Record start time
    start = std::chrono::high_resolution_clock::now();

    int i = 0;
    for (const auto& word_bitset : data_vec)
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
__global__ void find_ham1_GPU_ker(const unsigned int* subwords, unsigned int* pair_flags, const unsigned int subwords_per_pair_flags)
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
        if (hamming_distance && !(hamming_distance >> 1)) // true when hamming_distance == 1
        {
            flag_subword_offset = comparison_idx / UINT_BITSIZE;
            flag_in_subword = 1 << UINT_BITSIZE - 1 - comparison_idx % UINT_BITSIZE;
            pair_flags[word_idx * subwords_per_pair_flags + flag_subword_offset] |= flag_in_subword;
        }
    }

    delete[] word;
}

////////////////////////////////////////////////////////////////////////////////
// Counting kernel
__global__ void count_ones(unsigned int* d_data, size_t pair_flags_size)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= pair_flags_size)
        return;
    d_data[tid] = __popc(d_data[tid]);
}

////////////////////////////////////////////////////////////////////////////////
// finding pairs with hamming distance 1 on GPU
template<size_t N>
void find_ham1_GPU(thrust::device_vector<unsigned int>& d_subwords, \
    thrust::device_vector<unsigned int>& d_pair_flags, \
    thrust::host_vector<unsigned int>& h_pair_flags, size_t pair_flags_size, \
    const bool timeCount, const bool pairsOutput, const bool checkData, \
    const typename std::vector<std::bitset<N>>& data_vec, typename std::vector<std::pair<std::bitset<N>, std::bitset<N>>>& ham1_pairs)
{
    unsigned int threads = 512;
    unsigned int blocks = (unsigned int)std::ceil(DATA_SIZE / (double)threads);
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    unsigned int pairs_count_GPU = 0; //pairs_count = 0;
    const unsigned int subwords_per_pair_flags = pair_flags_size / DATA_SIZE;
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
    find_ham1_GPU_ker<<<dimGrid, dimBlock>>>(d_subwords_ptr, d_pair_flags_ptr, subwords_per_pair_flags);
    if (timeCount) cudaEventRecord(stop, 0);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();

    if (timeCount) cudaEventElapsedTime(&elapsed, start, stop);

    std::cout << "Finished!\n";
    if (timeCount) std::cout << "Elapsed time: " << elapsed << " ms\n";

    h_pair_flags = d_pair_flags;
    //for (size_t i = 0; i < pair_flags_size; ++i)
    //{
    //    pairs_count += __popcnt(h_pair_flags[i]);
    //}
    //std::cout << pairs_count << " pairs found\n\n";

    int threads2 = 512;
    int blocks2 = (unsigned int)std::ceil(pair_flags_size / (double)threads);
    dim3 dimBlock2(threads2, 1, 1);
    dim3 dimGrid2(blocks2, 1, 1);
    count_ones<<<dimGrid2, dimBlock2>>>(thrust::raw_pointer_cast(d_pair_flags.begin().base()), pair_flags_size);
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    pairs_count_GPU = thrust::reduce(d_pair_flags.begin(), d_pair_flags.end());
    d_pair_flags = h_pair_flags;
    
    std::cout << pairs_count_GPU << " pairs found\n\n";

    if (!pairs_count_GPU)
        process_pairs_from_flags<N>(h_pair_flags, pair_flags_size, data_vec, ham1_pairs, checkData, pairsOutput);
    else
        process_pairs_from_flags<N>(h_pair_flags, pair_flags_size, data_vec, ham1_pairs, checkData, false);

    if (timeCount) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    //thrust::fill(d_pair_flags.begin(), d_pair_flags.end(), 0);
}

////////////////////////////////////////////////////////////////////////////////
// pairs_flag to pairs output
template<size_t N>
void process_pairs_from_flags(thrust::host_vector<unsigned int>& h_pair_flags, size_t pair_flags_size, \
    const typename std::vector<std::bitset<N>>& data_vec, \
    const typename std::vector<std::pair<std::bitset<N>, std::bitset<N>>>& ham1_pairs, \
    const bool checkData, const bool pairsOutput)
{
    bool dataCorrect = true;
    const unsigned int subwords_per_word_flags = pair_flags_size / DATA_SIZE;

    if (pairsOutput) std::cout << "Pairs found:\n";

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
            int flags_set = __popcnt(word_flags[i]);
            int flag_pos = (i+1) * UINT_BITSIZE - 1;
            size_t j = 0;
            while (j < flags_set)
            {
                if (word_flags[i] % 2) {
                    if (checkData) {
                        std::pair<std::bitset<N>, std::bitset<N>> pair = std::make_pair<std::bitset<N>, std::bitset<N>>(std::bitset<N>(data_vec[word_idx].to_string()), std::bitset<N>(data_vec[flag_pos].to_string()));
                        if (std::end(ham1_pairs) == std::find(std::begin(ham1_pairs), std::end(ham1_pairs), pair)) {
                            std::cout << "No matching pair found in CPU Data" << std::endl;
                            dataCorrect = false;
                        }
                    }
                    else if (pairsOutput) std::cout << data_vec[word_idx] << " " << data_vec[flag_pos] << std::endl;
                    ++j;
                }
                word_flags[i] = word_flags[i] >> 1;
                --flag_pos;
            }
        }
        delete[] word_flags;
    }

    if (checkData && dataCorrect) std::cout << "GPU Data is consistent with CPU Data" << std::endl << std::endl;

    if (pairsOutput || !dataCorrect) std::cout << std::endl;
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
    std::vector<std::bitset<WORD_SIZE>> data_vec;
    std::vector<std::pair<std::bitset<WORD_SIZE>, std::bitset<WORD_SIZE>>> ham1_pairs;

    while (menu_choice != 5) {
        std::cout << "1. Generate Data" << std::endl;
        std::cout << "2. Save/Load Data" << std::endl;
        if (!data_vec.empty()) {
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
            std::cout << std::endl;
            while (menu_choice != 3)
            {
                std::cout << "1. Generate Random Data" << std::endl;
                std::cout << "2. Generate Predictable Data" << std::endl;
                std::cout << "3. Go Back" << std::endl;
                std::cout << "Choice: ";
                std::cin >> menu_choice;
                std::cout << std::endl;
                switch (menu_choice)
                {
                case 1:
                    if (!data_vec.empty())
                        data_vec.clear();
                    data_vec.reserve(DATA_SIZE);
                    if (!ham1_pairs.empty())
                        ham1_pairs.clear();
                    generate_random_data<WORD_SIZE, DATA_SIZE>(data_vec);
                    updated_data_GPU = false;
                    break;
                case 2:
                    if (!data_vec.empty())
                        data_vec.clear();
                    data_vec.reserve(DATA_SIZE);
                    if (!ham1_pairs.empty())
                        ham1_pairs.clear();
                    generate_predictable_data<WORD_SIZE, DATA_SIZE>(data_vec);
                    updated_data_GPU = false;
                    break;
                case 3:
                    break;
                default:
                    std::cout << "Please provide a valid choice" << std::endl << std::endl;
                    break;
                }
            }
            menu_choice = 1;
            break;
        case 2:
            std::cout << std::endl;
            while (menu_choice != 3)
            {
                std::cout << "1. Save Data" << std::endl;
                std::cout << "2. Load Data" << std::endl;
                std::cout << "3. Go Back" << std::endl;
                std::cout << "Choice: ";
                std::cin >> menu_choice;
                std::cout << std::endl;
                switch (menu_choice)
                {
                case 1:
                    save_data<WORD_SIZE, DATA_SIZE>("./words_data.csv", "./pairs_data.csv", data_vec, ham1_pairs);
                    break;
                case 2:
                    load_data<WORD_SIZE, DATA_SIZE>("./words_data.csv", "./pairs_data.csv", data_vec, ham1_pairs);
                    updated_data_GPU = false;
                    break;
                case 3:
                    break;
                default:
                    std::cout << "Please provide a valid choice" << std::endl << std::endl;
                    break;
                }
            }
            menu_choice = 2;
            break;
        case 3:
            if (!data_vec.empty()) {
                d_subwords = move_data_to_GPU<WORD_SIZE, DATA_SIZE>(data_vec);
                updated_data_GPU = true;
            }
            else
                std::cout << std::endl << "!!! Generate / Load Data before attempting to move the data to GPU !!!" << std::endl << std::endl;
            break;
        case 4:
            std::cout << std::endl;
            if (!data_vec.empty()) {
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
                    {
                        char c;
                        do {
                            std::cout << "Output pairs to console? (y/n):";
                            std::cin.clear();
                            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                            c = std::getc(stdin);
                            if (c == 'y' || c == 'Y') {
                                find_ham1<WORD_SIZE>(data_vec, ham1_pairs, true, true);
                                break;
                            }
                            else if (c == 'n' || c == 'N') {
                                find_ham1<WORD_SIZE>(data_vec, ham1_pairs, true, false);
                                break;
                            }
                            std::cout << "Please provide a valid choice" << std::endl;
                        } while (true);
                        break;
                    }
                    case 2:
                    {
                        bool out_to_console = false;
                        char c;
                        if (d_subwords.empty())
                            std::cout << std::endl << "!!! No Data on GPU !!!" << std::endl << std::endl;
                        else {
                            do {
                                std::cout << "Output pairs to console? (y/n):";
                                std::cin.clear();
                                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                                c = std::getc(stdin);
                                if (c == 'y' || c == 'Y') {
                                    out_to_console = true;
                                    break;
                                }
                                else if (c == 'n' || c == 'N') {
                                    out_to_console = false;
                                    break;
                                }
                                std::cout << "Please provide a valid choice" << std::endl;
                            } while (true);
                            do {
                                std::cout << "Check pairs against CPU? (y/n):";
                                std::cin.clear();
                                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                                c = std::getc(stdin);
                                if (c == 'y' || c == 'Y') {
                                    if (out_to_console) find_ham1_GPU<WORD_SIZE>(d_subwords, d_pair_flags, h_pair_flags, pair_flags_size, true, true, true, data_vec, ham1_pairs);
                                    else find_ham1_GPU<WORD_SIZE>(d_subwords, d_pair_flags, h_pair_flags, pair_flags_size, true, false, true, data_vec, ham1_pairs);
                                    break;
                                }
                                else if (c == 'n' || c == 'N') {
                                    if (out_to_console) find_ham1_GPU<WORD_SIZE>(d_subwords, d_pair_flags, h_pair_flags, pair_flags_size, true, true, false, data_vec, ham1_pairs);
                                    else find_ham1_GPU<WORD_SIZE>(d_subwords, d_pair_flags, h_pair_flags, pair_flags_size, true, false, false, data_vec, ham1_pairs);
                                    break;
                                }
                                std::cout << "Please provide a valid choice" << std::endl;
                            } while (true);
                        }
                        break;
                    }
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