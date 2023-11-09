#include <iostream>
#include <vector>
#include <chrono>

#include "onnx_wrapper.cpp"

int main(int argc, char **argv) 
{
    const std::string model_path = "/home/kmorgenstein/BDAI/onnx_wrapper/models/script.onnx";
    OnnxWrapper wrapper(model_path);

    std::vector<double> input(48); //Number of Inputs
    std::vector<float> time;
    std::vector<float> freq;
    int N = 100000; //Number of Test Iterations
    for (auto i = 0; i < N; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> output = wrapper.run(input);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
        time.push_back(duration.count());
        freq.push_back(1000000000 / duration.count()); //nsec to sec
    }
    auto const count = static_cast<float>(time.size());
    float mean = std::reduce(time.begin(), time.end()) / count / 1000000000; //nsec to sec
    float mean_hz = std::reduce(freq.begin(), freq.end()) / count;

    std::cout << "Mean Inference Time: " << mean << " s" << std::endl;
    std::cout << "Mean Hz: " << mean_hz << " Hz" << std::endl;

    return 0;
}