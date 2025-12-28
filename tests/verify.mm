#include <iostream>
#include <vector>
#include <algorithm>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

int main(int argc, char* argv[]) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Error: Metal is not supported on this device." << std::endl;
            return 1;
        }

        NSError* error = nil;
        NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
        NSString* libPath = [cwd stringByAppendingPathComponent:@"tests/test.metallib"];
        
        NSURL* libURL = [NSURL fileURLWithPath:libPath];
        id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&error];
        
        if (!library) {
            std::cerr << "Failed to load library: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLFunction> fn = [library newFunctionWithName:@"test_kernel"];
        if (!fn) {
            std::cerr << "Failed to find function 'test_kernel'." << std::endl;
            return 1;
        }

        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
        if (!pso) {
            std::cerr << "Failed to create pipeline state: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        const int count = 4;
        float rawData[count] = {10.0f, 20.0f, 30.0f, 40.0f};
        NSUInteger dataSize = count * sizeof(float);

        id<MTLBuffer> bufferA = [device newBufferWithBytes:rawData length:dataSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithLength:dataSize options:MTLResourceStorageModeShared];

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pso];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];

        // Shared Memory (4 floats)
        [encoder setThreadgroupMemoryLength:count*sizeof(float) atIndex:0];

        MTLSize gridSize = MTLSizeMake(count, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(count, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        float* resultPtr = (float*)bufferB.contents;
        std::cout << "Results:" << std::endl;
        bool success = true;
        for (int i = 0; i < count; ++i) {
            int neighbor_idx = i ^ 1;
            float expected = rawData[neighbor_idx];
            
            std::cout << "[" << i << "] In: " << rawData[i] 
                      << " -> Out: " << resultPtr[i]
                      << " (Exp: " << expected << ")";
            
            if (resultPtr[i] != expected) {
                std::cout << " FAIL";
                success = false;
            } else {
                std::cout << " OK";
            }
            std::cout << std::endl;
        }
        
        return success ? 0 : 1;
    }
}
