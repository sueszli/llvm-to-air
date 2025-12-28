#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <iostream>
#include <vector>
#include <algorithm>

#include "metal-cpp/Foundation/Foundation.hpp"
#include "metal-cpp/Metal/Metal.hpp"

int main() {
    // 1. Get the Metal Device
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Error: Metal is not supported on this device." << std::endl;
        return 1;
    }
    std::cout << "Device: " << device->name()->utf8String() << std::endl;

    // 2. Load the .metallib
    NS::Error* error = nullptr;
    
    // Get current working directory using NS::FileManager is a bit tricky in pure C++ binding 
    // without autorelease pool context sometimes, but let's stick to standard C++ filesystem or simple relative path
    // The example executed from the dir, so relative path "shader.metallib" is fine.
    // However, to match the original closely, let's use C strings for path if possible, 
    // or just construct NS::String.
    
    // We need an AutoreleasePool for some NS objects if they are autoreleased.
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    NS::String* libPath = NS::String::string("shader.metallib", NS::UTF8StringEncoding);
    MTL::Library* library = device->newLibrary(libPath, &error);
    
    if (!library) {
        std::cerr << "Failed to load library 'shader.metallib': " 
                  << error->localizedDescription()->utf8String() << std::endl;
        return 1;
    }

    // 3. Get the kernel function "add"
    NS::String* fnName = NS::String::string("add", NS::UTF8StringEncoding);
    MTL::Function* addFn = library->newFunction(fnName);
    if (!addFn) {
        std::cerr << "Failed to find function 'add' in library." << std::endl;
        return 1;
    }

    // 4. Create Compute Pipeline State
    MTL::ComputePipelineState* pso = device->newComputePipelineState(addFn, &error);
    if (!pso) {
        std::cerr << "Failed to create pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
        return 1;
    }

    // 5. Prepare Data
    const int count = 4;
    float rawData[count] = {10.0f, 20.0f, 30.0f, 40.0f};
    NS::UInteger dataSize = count * sizeof(float);

    MTL::Buffer* bufferA = device->newBuffer(rawData, dataSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferB = device->newBuffer(dataSize, MTL::ResourceStorageModeShared);

    // 6. Encode Commands
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

    // 7. Execute and Wait
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    // 8. Verify
    float* resultPtr = (float*)bufferB->contents();
    std::cout << "Results:" << std::endl;
    bool success = true;
    for (int i = 0; i < count; ++i) {
        float expected = rawData[i] + 1.0f;
        std::cout << "[" << i << "] Input: " << rawData[i] 
                  << " -> Output: " << resultPtr[i];
        if (resultPtr[i] != expected) {
            std::cout << " (FAIL: expected " << expected << ")";
            success = false;
        } else {
            std::cout << " (OK)";
        }
        std::cout << std::endl;
    }
    
    // Cleanup manually as we are not using smart pointers for everything here (raw raw refs)
    // Note: In a real app we'd use NS::SharedPtr or similar. 
    // Since we are exiting, OS will reclaim, but let's release explicit objects to be good citizens
    bufferA->release();
    bufferB->release();
    pso->release();
    addFn->release();
    library->release();
    // commandBuffer is autoreleased, do not manually release.
    commandQueue->release();
    device->release();
    
    pool->release();

    if (success) {
        std::cout << "SUCCESS: Kernel execution verified." << std::endl;
        return 0;
    } else {
        std::cerr << "FAILURE: Results did not match." << std::endl;
        return 1;
    }
}
