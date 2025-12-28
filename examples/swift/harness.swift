import Metal
import Foundation

func main() -> Int32 {
    guard let device = MTLCreateSystemDefaultDevice() else {
        fputs("error: metal is not supported on this device.\n", stderr)
        return 1
    }
    print("device: \(device.name)")

    let fileManager = FileManager.default
    let cwd = fileManager.currentDirectoryPath
    let libPath = URL(fileURLWithPath: cwd).appendingPathComponent("shader.metallib")
    
    let library: MTLLibrary
    do {
        library = try device.makeLibrary(URL: libPath)
    } catch {
        fputs("failed to load library '\(libPath.path)': \(error.localizedDescription)\n", stderr)
        return 1
    }

    guard let addFn = library.makeFunction(name: "add") else {
        fputs("failed to find function 'add' in library.\n", stderr)
        return 1
    }

    let pso: MTLComputePipelineState
    do {
        pso = try device.makeComputePipelineState(function: addFn)
    } catch {
        fputs("failed to create pipeline state: \(error.localizedDescription)\n", stderr)
        return 1
    }

    let count = 4
    // input: {10, 20, 30, 40}
    var rawData: [Float] = [10.0, 20.0, 30.0, 40.0]
    let dataSize = count * MemoryLayout<Float>.size

    guard let bufferA = device.makeBuffer(bytes: &rawData, length: dataSize, options: .storageModeShared),
          let bufferB = device.makeBuffer(length: dataSize, options: .storageModeShared) else {
        fputs("failed to create buffers.\n", stderr)
        return 1
    }

    guard let commandQueue = device.makeCommandQueue(),
          let commandBuffer = commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
        fputs("failed to create command objects.\n", stderr)
        return 1
    }

    encoder.setComputePipelineState(pso)
    encoder.setBuffer(bufferA, offset: 0, index: 0)
    encoder.setBuffer(bufferB, offset: 0, index: 1)

    let gridSize = MTLSizeMake(count, 1, 1)
    let threadsPerGroupVal = min(pso.maxTotalThreadsPerThreadgroup, count)
    let threadGroupSize = MTLSizeMake(threadsPerGroupVal, 1, 1)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // verify
    let resultPtr = bufferB.contents().bindMemory(to: Float.self, capacity: count)
    print("results:")
    var success = true
    for i in 0..<count {
        let input = rawData[i]
        let output = resultPtr[i]
        let expected = input + 1.0
        
        print("[\(i)] input: \(input) -> output: \(output)", terminator: "")
        
        if output != expected {
            print(" (FAIL: expected \(expected))")
            success = false
        } else {
            print(" (OK)")
        }
    }

    if success {
        print("SUCCESS: kernel execution verified.")
        return 0
    } else {
        fputs("FAILURE: results did not match.\n", stderr)
        return 1
    }
}

exit(main())
