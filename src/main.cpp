#include "onnx_wrapper.cpp"

int main(int argc, char **argv) 
{
    OnnxWrapper wrapper;
    const std::string& config_path = "/home/kmorgenstein/BDAI/onnx_wrapper/src/onnx_config.yaml";
    wrapper.initialize(config_path);
    return 0;
}