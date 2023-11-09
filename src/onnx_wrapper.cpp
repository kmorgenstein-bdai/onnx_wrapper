#include "onnx_wrapper.h"

// Constructor
OnnxWrapper::OnnxWrapper(const std::string model_path)
{
    // Create ORT Environment
    std::string instanceName{"ONNX Wrapper"};
    envPtr = std::make_shared<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());

    // Create ORT Session
    Ort::SessionOptions sessionOptions;
    // sessionOptions.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{}); //Enable CUDA (currently unused)
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

    // Input Info
    //size_t numInputNodes = sessionPtr->GetInputCount(); //Number of Input Nodes (currently unused)
    Ort::TypeInfo inputTypeInfo = sessionPtr->GetInputTypeInfo(0); //Input Type Info
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo(); //Input Tensor Info
    //ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType(); //Input Type (currently unused)
    inputDims = inputTensorInfo.GetShape(); //Input Dims
    if (inputDims.at(0) != 1){inputDims.at(0) = 1;} //Set Batch Size = 1

    // Output Info
    //size_t numOutputNodes = sessionPtr->GetOutputCount(); //Number of Output Nodes (currently unused)
    Ort::TypeInfo outputTypeInfo = sessionPtr->GetOutputTypeInfo(0); //Output Type Info
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo(); //Output Tensor Info
    //ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType(); //Output Type (currently unused)
    outputDims = outputTensorInfo.GetShape(); //Output Dims
    if (outputDims.at(0) != 1){outputDims.at(0) = 1;} //Set Batch Size = 1

    prettyPrint("[OnnxWrapper] Policy Initialized", printColors::green);
    return;
}

std::vector<double> OnnxWrapper::run(std::vector<double> inputData)
{
    // Memory Info
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Inputs
    Ort::AllocatedStringPtr inputNamePtr = sessionPtr->GetInputNameAllocated(0, allocator); //Input Name Ptr
    std::vector<const char*> inputNames{inputNamePtr.release()}; //Input Name
    size_t inputTensorSize = vectorProduct(inputDims); //Input Tensor Size
    std::vector<float> inputTensorValues(inputTensorSize); //Create Input Values Tensor
    inputTensorValues.assign(inputData.begin(), inputData.end()); //Fill Input Values Tensor
    std::vector<Ort::Value> inputTensors; //Create Input ORT Tensor
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(), inputDims.size())); //Fill Input ORT Tensor
    
    // Outputs
    Ort::AllocatedStringPtr outputNamePtr = sessionPtr->GetOutputNameAllocated(0, allocator); //Output Name Ptr
    std::vector<const char*> outputNames{outputNamePtr.release()}; //Output Name
    size_t outputTensorSize = vectorProduct(outputDims); //Output Tensor Size
    std::vector<float> outputTensorValues(outputTensorSize); //Create Output Value Tensor
    std::vector<Ort::Value> outputTensors; //Create Output ORT Tensor
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data(), outputTensorSize, outputDims.data(), outputDims.size())); //Fill Output ORT Tensor

    // Inference
    sessionPtr->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), 1, outputNames.data(), outputTensors.data(), 1); //1 Input + 1 Output

    // Post Processing
    std::vector<double> output(outputTensorSize); //Create Output Values Vector
    output.assign(outputTensorValues.begin(), outputTensorValues.end()); //Fill Output Values Vector

    return output;
}