#include "onnx_wrapper.h"
#include <ParamHandler.hpp> 

// Constructor
OnnxWrapper::OnnxWrapper()
{
    initialize();
    prettyPrint("[OnnxWrapper] OnnxWrapper Constructed", printColors::green);
}

// Initializer
void OnnxWrapper::initialize()
{
    const std::string& config_path = "/home/kmorgenstein/BDAI/inference_test/onnx_config.yaml";
    ParamHandler paramHandler(config_path);
    if (!paramHandler.fileOpenedSuccessfully())
    {
        prettyPrint("[OnnxWrapper] Bad configuration file", printColors::red);
        exit(1);
    }

    //std::vector<std::string> keys = paramHandler.getKeys();
    //for (auto& key : keys)
    //{
    //    if (key == "model"){std::string output; paramHandler.getValue(key, output); model = output; continue;}
    //    if (key == "input_dim"){double output; paramHandler.getValue(key, output); input_dim = (int) output; continue;}
    //    if (key == "output_dim"){double output; paramHandler.getValue(key, output); output_dim = (int) output; continue;}
    //}
    std::cout << model << std::endl;
    std::cout << input_dim << std::endl;
    std::cout << output_dim << std::endl;

    prettyPrint("[OnnxWrapper] OnnxWrapper Initialized", printColors::green);
    return;
}