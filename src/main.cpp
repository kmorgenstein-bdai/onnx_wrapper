#include "onnx_wrapper.cpp"

int main(int argc, char **argv) 
{
    OnnxWrapper wrapper;
    const std::string model_path = "/home/kmorgenstein/BDAI/onnx_wrapper/models/script.onnx";
    wrapper.initialize(model_path);
    return 0;
}