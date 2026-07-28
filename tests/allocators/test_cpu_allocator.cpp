#include <gtest/gtest.h>
#include <cstdint>
#include <thread>
#include <vector>

#include "nova/allocators/allocator_registry.hpp"

TEST(CPUAllocatorTest, AllocateMemory)
{
    auto& allocator =
        nova::AllocatorRegistry::cpu();

    auto block =
        allocator.allocate(1024);

    EXPECT_NE(block.data, nullptr);

    EXPECT_EQ(block.bytes, 1024);

    allocator.deallocate(block);
}

TEST(CPUAllocatorTest, Alignment64)
{
    auto& allocator =
        nova::AllocatorRegistry::cpu();

    auto block =
        allocator.allocate(4096, 64);

    auto address =
        reinterpret_cast<std::uintptr_t>(block.data);

    EXPECT_EQ(address % 64, 0);

    allocator.deallocate(block);
}


TEST(CPUAllocatorTest, ManyAllocations)
{
    auto& allocator =
        nova::AllocatorRegistry::cpu();

    for(int i = 0; i < 10000; i++)
    {
        auto block =
            allocator.allocate(512);

        EXPECT_NE(block.data, nullptr);

        allocator.deallocate(block);
    }
}


TEST(CPUAllocatorTest, WriteRead)
{
    auto& allocator =
        nova::AllocatorRegistry::cpu();

    auto block =
        allocator.allocate(100 * sizeof(float));

    auto* ptr =
        static_cast<float*>(block.data);

    for(int i = 0; i < 100; i++)
        ptr[i] = static_cast<float>(i);

    for(int i = 0; i < 100; i++)
        EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(i));

    allocator.deallocate(block);
}

TEST(CPUAllocatorTest, MultipleAlignments)
{
    auto& allocator =
        nova::AllocatorRegistry::cpu();

    constexpr std::size_t alignments[] =
    {
        16,
        32,
        64,
        128,
        256
    };

    for(auto alignment : alignments)
    {
        auto block =
            allocator.allocate(4096, alignment);

        auto addr =
            reinterpret_cast<std::uintptr_t>(block.data);

        EXPECT_EQ(addr % alignment, 0);

        allocator.deallocate(block);
    }
}

TEST(CPUAllocatorTest, MultiThread)
{
    auto& allocator =
        nova::AllocatorRegistry::cpu();

    auto worker =
    [&]()
    {
        for(int i = 0; i < 5000; i++)
        {
            auto block =
                allocator.allocate(512);

            EXPECT_NE(block.data, nullptr);

            allocator.deallocate(block);
        }
    };

    std::vector<std::thread> threads;

    for(int i = 0; i < 16; i++)
        threads.emplace_back(worker);

    for(auto& t : threads)
        t.join();
}