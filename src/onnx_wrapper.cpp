#include "onnx_wrapper.h"
#include <ParamHandler.hpp> 

// Constructor
OnnxWrapper::OnnxWrapper()
{
    prettyPrint("[OnnxWrapper] OnnxWrapper Created", printColors::green);
}

// Initializer
void OnnxWrapper::initialize(const std::string& config_path)
{
    ParamHandler paramHandler(config_path);
    if (!paramHandler.fileOpenedSuccessfully())
    {
        prettyPrint("[OnnxWrapper] Bad configuration file", printColors::red);
        exit(1);
    }

    // Read Keys from YAML file
    std::vector<std::string> keys = paramHandler.getKeys();
    for (auto& key : keys)
    {
        if (key == "model"){std::string output; paramHandler.getValue(key, output); model = output; continue;}
    }

    // test that yaml is read correctly - delete once it works
    std::cout << model << std::endl;

    // Create ORT Environment
    std::string instanceName{"ONNX Wrapper"};
    envPtr = std::make_shared<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());

    // Create ORT Session
    Ort::SessionOptions sessionOptions;
    // sessionOptions.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{}); //Enable CUDA
    sessionOptions.SetIntraOpNumThreads(1); //Set Threads
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); //Enable Optimizations
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
    // removals) 
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    sessionPtr = std::make_shared<Ort::Session>(*envPtr, model.c_str(), sessionOptions); //Load the ONNX Model

    // Create Allocator
    Ort::AllocatorWithDefaultOptions allocator;

    // Input Info
    size_t numInputNodes = sessionPtr->GetInputCount(); //Number of Input Nodes
    Ort::TypeInfo inputTypeInfo = sessionPtr->GetInputTypeInfo(0); //Input Type Info
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo(); //Input Tensor Info
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType(); //Input Type
    inputDims = inputTensorInfo.GetShape(); //Input Dims
    if (inputDims.at(0) != 1){inputDims.at(0) = 1;} //Set Batch Size = 1

    //Output Info
    size_t numOutputNodes = sessionPtr->GetOutputCount(); //Number of Output Nodes
    Ort::TypeInfo outputTypeInfo = sessionPtr->GetOutputTypeInfo(0); //Output Type Info
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo(); //Output Tensor Info
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType(); //Output Type
    outputDims = outputTensorInfo.GetShape(); //Output Dims
    if (outputDims.at(0) != 1){outputDims.at(0) = 1;} //Set Batch Size = 1

    prettyPrint("[OnnxWrapper] Policy Initialized", printColors::green);
    return;
}

void OnnxWrapper::run() //need to add input and output data types
{

}