#include <iostream>
#include <vector>
#include <algorithm>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "error: metal is not supported on this device." << std::endl;
            return 1;
        }
        std::cout << "device: " << [device.name UTF8String] << std::endl;

        NSError* error = nil;
        NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
        NSString* libPath = [cwd stringByAppendingPathComponent:@"shader.metallib"];
        
        NSURL* libURL = [NSURL fileURLWithPath:libPath];
        id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&error];
        
        if (!library) {
            std::cerr << "failed to load library '" << [libPath UTF8String] << "': " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        // get the kernel function "add"
        id<MTLFunction> addFn = [library newFunctionWithName:@"add"];
        if (!addFn) {
            std::cerr << "failed to find function 'add' in library." << std::endl;
            return 1;
        }

        // create compute pipeline state
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:addFn error:&error];
        if (!pso) {
            std::cerr << "failed to create pipeline state: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        const int count = 4;
        // input: {10, 20, 30, 40}
        float rawData[count] = {10.0f, 20.0f, 30.0f, 40.0f};
        NSUInteger dataSize = count * sizeof(float);

        // buffer(0) is input A
        // buffer(1) is output B
        id<MTLBuffer> bufferA = [device newBufferWithBytes:rawData length:dataSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithLength:dataSize options:MTLResourceStorageModeShared];

        // encode commands
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pso];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];

        // dispatch (for simple 1D, we can use dispatchThreads)
        MTLSize gridSize = MTLSizeMake(count, 1, 1);
        NSUInteger threadsPerGroupVal = std::min((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)count);
        MTLSize threadGroupSize = MTLSizeMake(threadsPerGroupVal, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        // execute and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // verify
        float* resultPtr = (float*)bufferB.contents;
        std::cout << "results:" << std::endl;
        bool success = true;
        for (int i = 0; i < count; ++i) {
            float expected = rawData[i] + 1.0f;
            std::cout << "[" << i << "] input: " << rawData[i] << " -> output: " << resultPtr[i];
            if (resultPtr[i] != expected) {
                std::cout << " (FAIL: expected " << expected << ")";
                success = false;
            } else {
                std::cout << " (OK)";
            }
            std::cout << std::endl;
        }
        
        if (success) {
            std::cout << "SUCCESS: kernel execution verified." << std::endl;
        } else {
            std::cerr << "FAILURE: results did not match." << std::endl;
            return 1;
        }
    }
    return 0;
}
