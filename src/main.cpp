#include "onnx_wrapper.cpp"

int main(int argc, char **argv) 
{
    const std::string model_path = "/home/kmorgenstein/BDAI/onnx_wrapper/models/script.onnx";
    OnnxWrapper wrapper(model_path);
    return 0;
}