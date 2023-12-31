# ifndef ONNX_WRAPPER_H
# define ONNX_WRAPPER_H

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <cstddef>
#include <onnxruntime_cxx_api.h>

template <typename T>
size_t vectorProduct(const std::vector<T>& v)
{
  return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

enum printColors {black, red, green, blue, yellow, magenta, cyan, white};

template <typename T>
void prettyPrint(T input, int color = printColors::black)
{
  switch (color)
  {
    case printColors::black:
      std::cout << input << std::endl;
      break;
    case printColors::red:
      std::cout << "\033[1;31m" << input << "\033[0m\n";
      break;
    case printColors::green:
      std::cout << "\033[1;32m" << input << "\033[0m\n";
      break;
    case printColors::yellow:
      std::cout << "\033[1;33m" << input << "\033[0m\n";
      break;
    case printColors::blue:
      std::cout << "\033[1;34m" << input << "\033[0m\n";
      break;
    case printColors::magenta:
      std::cout << "\033[1;35m" << input << "\033[0m\n";
      break;
    case printColors::cyan:
      std::cout << "\033[1;36m" << input << "\033[0m\n";
      break;
    case printColors::white:
      std::cout << "\033[1;37m" << input << "\033[0m\n";
      break;
  }
}

class OnnxWrapper {
    public:
        OnnxWrapper(const std::string modelPath);
        std::vector<double> run(std::vector<double> inputData);
    private:
        // ORT Environment
        std::shared_ptr<Ort::Env> envPtr;

        // Session
        std::shared_ptr<Ort::Session> sessionPtr;

        // Allocator
        Ort::AllocatorWithDefaultOptions allocator;

        // Inputs
        std::vector<int64_t> inputDims;

        // Outputs
        std::vector<int64_t> outputDims;
};

# endif