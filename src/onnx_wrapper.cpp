#include "onnx_wrapper.h"


// Constructor
OnnxWrapper::OnnxWrapper(const std::string model_path)
{
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
    sessionPtr = std::make_shared<Ort::Session>(*envPtr, model_path.c_str(), sessionOptions); //Load the ONNX Model

    // Create Allocator
    //Ort::AllocatorWithDefaultOptions allocator;

    // Input Info
    size_t numInputNodes = sessionPtr->GetInputCount(); //Number of Input Nodes
    Ort::TypeInfo inputTypeInfo = sessionPtr->GetInputTypeInfo(0); //Input Type Info
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo(); //Input Tensor Info
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType(); //Input Type
    inputDims = inputTensorInfo.GetShape(); //Input Dims
    if (inputDims.at(0) != 1){inputDims.at(0) = 1;} //Set Batch Size = 1

    // Output Info
    size_t numOutputNodes = sessionPtr->GetOutputCount(); //Number of Output Nodes
    Ort::TypeInfo outputTypeInfo = sessionPtr->GetOutputTypeInfo(0); //Output Type Info
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo(); //Output Tensor Info
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType(); //Output Type
    outputDims = outputTensorInfo.GetShape(); //Output Dims
    if (outputDims.at(0) != 1){outputDims.at(0) = 1;} //Set Batch Size = 1

    prettyPrint("[OnnxWrapper] Policy Initialized", printColors::green);
    return;
}

void OnnxWrapper::run(std::vector<double> inputData)
{
    // Inputs
    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<float> inputTensorValues(inputTensorSize);
    inputTensorValues.assign(inputData.begin(), inputData.end());
    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(), inputDims.size()));
    Ort::AllocatedStringPtr inputNamePtr = sessionPtr->GetInputNameAllocated(0, allocator); //Input Name

    // Outputs
    size_t outputTensorSize = vectorProduct(outputDims);
    std::vector<float> outputTensorValues(outputTensorSize);
    std::vector<Ort::Value> outputTensors;
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data(), outputTensorSize, outputDims.data(), outputDims.size()));
    Ort::AllocatedStringPtr outputNamePtr = sessionPtr->GetOutputNameAllocated(0, allocator); //Input Name

    // Inference
    std::vector<const char*> inputNames{inputNamePtr.release()};
    std::vector<const char*> outputNames{outputNamePtr.release()};
    sessionPtr->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), 1, outputNames.data(), outputTensors.data(), 1);
}