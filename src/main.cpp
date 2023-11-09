#include <iostream>
#include <vector>
#include <chrono>

#include "onnx_wrapper.cpp"

int main(int argc, char **argv) 
{
    std::string filePath {__FILE__};
    const std::string modelPath = filePath.substr(0, filePath.rfind("/")) + "/models/dummy.onnx";
    OnnxWrapper wrapper(modelPath);

    std::vector<double> input(48); //Number of Inputs
    std::vector<float> time;
    std::vector<float> freq;
    int N = 1000; //Number of Test Iterations
    prettyPrint("[OnnxWrapper] Running Inference for " + std::to_string(N) + " Iterations", printColors::yellow);
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

    prettyPrint("[OnnxWrapper] Mean Inference Time: " + std::to_string(mean) + " s", printColors::yellow);
    prettyPrint("[OnnxWrapper] Mean Inference Frequency: " + std::to_string(mean_hz) + " Hz", printColors::yellow);

    return 0;
}