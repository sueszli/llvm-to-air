#include <iostream>
#include <vector>
#include <algorithm>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

int main() {
    @autoreleasepool {
        // 1. Get the Metal Device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Error: Metal is not supported on this device." << std::endl;
            return 1;
        }
        std::cout << "Device: " << [device.name UTF8String] << std::endl;

        // 2. Load the .metallib
        // We use full path to be safe, assuming it's in the current working directory
        NSError* error = nil;
        NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
        NSString* libPath = [cwd stringByAppendingPathComponent:@"shader.metallib"];
        
        NSURL* libURL = [NSURL fileURLWithPath:libPath];
        id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&error];
        
        if (!library) {
            std::cerr << "Failed to load library '" << [libPath UTF8String] << "': " 
                      << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        // 3. Get the kernel function "add"
        id<MTLFunction> addFn = [library newFunctionWithName:@"add"];
        if (!addFn) {
            std::cerr << "Failed to find function 'add' in library." << std::endl;
            return 1;
        }

        // 4. Create Compute Pipeline State
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:addFn error:&error];
        if (!pso) {
            std::cerr << "Failed to create pipeline state: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        // 5. Prepare Data
        const int count = 4;
        // Input: {10, 20, 30, 40}
        float rawData[count] = {10.0f, 20.0f, 30.0f, 40.0f};
        NSUInteger dataSize = count * sizeof(float);

        // buffer(0) is input A
        // buffer(1) is output B
        id<MTLBuffer> bufferA = [device newBufferWithBytes:rawData length:dataSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithLength:dataSize options:MTLResourceStorageModeShared];

        // 6. Encode Commands
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pso];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];

        // Dispatch
        // For simple 1D, we can use dispatchThreads provided the device supports it (Apple Silicon does)
        MTLSize gridSize = MTLSizeMake(count, 1, 1);
        NSUInteger threadsPerGroupVal = std::min((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)count);
        MTLSize threadGroupSize = MTLSizeMake(threadsPerGroupVal, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        // 7. Execute and Wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // 8. Verify
        float* resultPtr = (float*)bufferB.contents;
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
        
        if (success) {
            std::cout << "SUCCESS: Kernel execution verified." << std::endl;
        } else {
            std::cerr << "FAILURE: Results did not match." << std::endl;
            return 1;
        }
    }
    return 0;
}
