static const MemoryScope memoryScope = MemoryScope::Workgroup;
static const int alignment = 4;

uniform RWStructuredBuffer<int> buffer;
[numthreads(32, 32, 1)]
void computeMain()
{
    for (int i = 0; i < 1000000; i++)
    {
        int currentData = coherentLoad(&buffer[i + 1], alignment, memoryScope);
        coherentStore(&buffer[i], currentData, alignment, memoryScope);
    }
}