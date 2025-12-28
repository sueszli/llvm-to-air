#include <cassert>
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <iostream>
#include <vector>
#include <algorithm>

#include "metal-cpp/Foundation/Foundation.hpp"
#include "metal-cpp/Metal/Metal.hpp"

int main() {
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "ERROR: metal is not supported on this device." << std::endl;
        return 1;
    }
    std::cout << "device: " << device->name()->utf8String() << std::endl;

    // load the .metallib
    NS::Error* error = nullptr;
    
    // we need an autorelease pool for some NS objects if they are autoreleased.
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    NS::String* libPath = NS::String::string("shader.metallib", NS::UTF8StringEncoding);
    MTL::Library* library = device->newLibrary(libPath, &error);
    assert(library && "failed to load library 'shader.metallib'");

    // get the kernel function "add"
    NS::String* fnName = NS::String::string("add", NS::UTF8StringEncoding);
    MTL::Function* addFn = library->newFunction(fnName);
    assert(addFn && "failed to find function 'add'");

    // create compute pipeline state
    MTL::ComputePipelineState* pso = device->newComputePipelineState(addFn, &error);
    assert(pso && "failed to create pipeline state");

    const int count = 4;
    float rawData[count] = {10.0f, 20.0f, 30.0f, 40.0f};
    NS::UInteger dataSize = count * sizeof(float);

    MTL::Buffer* bufferA = device->newBuffer(rawData, dataSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferB = device->newBuffer(dataSize, MTL::ResourceStorageModeShared);

    // encode commands
    MTL::CommandQueue* commandQueue = device->newCommandQueue();
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

    encoder->setComputePipelineState(pso);
    encoder->setBuffer(bufferA, 0, 0);
    encoder->setBuffer(bufferB, 0, 1);

    MTL::Size gridSize = MTL::Size::Make(count, 1, 1);
    NS::UInteger threadsPerGroupVal = std::min((NS::UInteger)pso->maxTotalThreadsPerThreadgroup(), (NS::UInteger)count);
    MTL::Size threadGroupSize = MTL::Size::Make(threadsPerGroupVal, 1, 1);

    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();

    // execute and wait
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    // verify
    float* resultPtr = (float*)bufferB->contents();
    std::cout << "results:" << std::endl;
    bool success = true;
    for (int i = 0; i < count; ++i) {
        float expected = rawData[i] + 1.0f;
        std::cout << "[" << i << "] input: " << rawData[i] << " -> output: " << resultPtr[i];
        assert(resultPtr[i] == expected);
        std::cout << std::endl;
    }
    
    // note: in production code we'd use NS::SharedPtr or similar. 
    bufferA->release();
    bufferB->release();
    pso->release();
    addFn->release();
    library->release();
    commandQueue->release();
    device->release();

    pool->release();

    assert(success && "results did not match");
    std::cout << "SUCCESS: kernel execution verified." << std::endl;
    return 0;
}
