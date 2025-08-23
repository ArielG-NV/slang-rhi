#include "testing.h"
#include "core/timer.h"

using namespace rhi;
using namespace rhi::testing;

static Result loadProgram(
    IDevice* device,
    ComPtr<IShaderProgram>& outShaderProgram,
    const char* shaderModuleName,
    const char* entryPointName,
    slang::ProgramLayout*& slangReflection)
{
    ComPtr<slang::ISession> slangSession;
    SLANG_RETURN_ON_FAIL(device->getSlangSession(slangSession.writeRef()));
    ComPtr<slang::IBlob> diagnosticsBlob;
    slang::IModule* module = slangSession->loadModule(shaderModuleName, diagnosticsBlob.writeRef());
    diagnoseIfNeeded(diagnosticsBlob);
    if (!module)
        return SLANG_FAIL;

    ComPtr<slang::IEntryPoint> computeEntryPoint;
    SLANG_RETURN_ON_FAIL(module->findEntryPointByName(entryPointName, computeEntryPoint.writeRef()));

    std::vector<slang::IComponentType*> componentTypes;
    componentTypes.push_back(module);
    componentTypes.push_back(computeEntryPoint);

    ComPtr<slang::IComponentType> composedProgram;
    Result result = slangSession->createCompositeComponentType(
        componentTypes.data(),
        componentTypes.size(),
        composedProgram.writeRef(),
        diagnosticsBlob.writeRef()
    );
    diagnoseIfNeeded(diagnosticsBlob);
    SLANG_RETURN_ON_FAIL(result);

    ComPtr<slang::IComponentType> linkedProgram;
    result = composedProgram->link(linkedProgram.writeRef(), diagnosticsBlob.writeRef());
    diagnoseIfNeeded(diagnosticsBlob);
    SLANG_RETURN_ON_FAIL(result);

    slangReflection = linkedProgram->getLayout();
    outShaderProgram = device->createShaderProgram(linkedProgram);
    return outShaderProgram ? SLANG_OK : SLANG_FAIL;
}

double dispatchAndTime(IDevice* device, ComPtr<IShaderProgram> shaderProgram, slang::ProgramLayout*& slangReflection)
{
    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));

    ComPtr<IBuffer> buffer;
    {
        const int numberCount = 10000000;
        int32_t* initialDataBuffer = new int32_t[numberCount];
        BufferDesc bufferDesc = {};
        bufferDesc.size = numberCount * sizeof(int32_t);
        bufferDesc.format = Format::Undefined;
        bufferDesc.elementSize = sizeof(int32_t);
        bufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopyDestination |
                           BufferUsage::CopySource;
        bufferDesc.defaultState = ResourceState::UnorderedAccess;
        bufferDesc.memoryType = MemoryType::DeviceLocal;

        REQUIRE_CALL(device->createBuffer(bufferDesc, (void*)initialDataBuffer, buffer.writeRef()));
        delete[] initialDataBuffer;
    }

    // We have done all the set up work, now it is time to start recording a command buffer for
    // GPU execution.
    {
        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();

        auto passEncoder = commandEncoder->beginComputePass();
        auto rootObject = passEncoder->bindPipeline(pipeline);
        ShaderCursor entryPointCursor(rootObject);
        entryPointCursor["buffer"].setBinding(buffer);
        Timer timer;
        TimePoint start = timer.now();
        passEncoder->dispatchCompute(1, 1, 1);
        passEncoder->end();
        queue->submit(commandEncoder->finish());
        queue->waitOnHost();
        TimePoint end = timer.now();
        double time = Timer::delta(start, end);
        return time;
    }
}

GPU_TEST_CASE("coherent-perf-comparison", Vulkan)
{
    ComPtr<IShaderProgram> shaderProgramWarmup;
    slang::ProgramLayout* slangReflectionWarmup = nullptr;
    REQUIRE_CALL(loadProgram(
        device,
        shaderProgramWarmup,
        "test-coherent-perf-warmup.slang",
        "computeMain",
        slangReflectionWarmup
    ));

    ComPtr<IShaderProgram> shaderProgramDevice;
    slang::ProgramLayout* slangReflectionDevice = nullptr;
    REQUIRE_CALL(loadProgram(
        device,
        shaderProgramDevice,
        "test-coherent-perf-device.slang",
        "computeMain",
        slangReflectionDevice
    )
    );

    ComPtr<IShaderProgram> shaderProgramWorkgroup;
    slang::ProgramLayout* slangReflectionWorkgroup = nullptr;
    REQUIRE_CALL(loadProgram(
        device,
        shaderProgramWorkgroup,
        "test-coherent-perf-workgroup.slang",
        "computeMain",
        slangReflectionWorkgroup
    )
    );

    // Note: not using link-time-constants due to bug(?) causing them to fail
    const int trials = 3;
    double totalDeviceTime = 0;
    double totalWorkgroupTime = 0;
    for (auto i = 0; i < trials; i++)
    {
        dispatchAndTime(device, shaderProgramWarmup, slangReflectionWarmup);

        totalDeviceTime += 1.0f / (double)trials * dispatchAndTime(device, shaderProgramDevice, slangReflectionDevice);

        dispatchAndTime(device, shaderProgramWarmup, slangReflectionWarmup);

        totalWorkgroupTime +=
            1.0f / (double)trials * dispatchAndTime(device, shaderProgramWorkgroup, slangReflectionWorkgroup);
    }
    printf("\n Device, Time: %f \n", totalDeviceTime);
    printf("\n Workgroup, Time: %f \n", totalWorkgroupTime);
}
