#include <gtest/gtest.h>

#include "nova/storage/storage.hpp"
#include "nova/allocators/allocator_registry.hpp"

TEST(StorageTest, DefaultConstructor)
{
    nova::Storage storage;

    EXPECT_TRUE(storage.empty());
    EXPECT_EQ(storage.bytes(), 0);
    EXPECT_EQ(storage.data(), nullptr);
}

TEST(StorageTest, AllocateStorage)
{
    auto& allocator = nova::AllocatorRegistry::cpu();

    nova::Storage storage(1024, allocator);

    EXPECT_FALSE(storage.empty());
    EXPECT_EQ(storage.bytes(), 1024);
    EXPECT_NE(storage.data(), nullptr);
}

TEST(StorageTest, ReadWriteMemory)
{
    auto& allocator = nova::AllocatorRegistry::cpu();

    nova::Storage storage(100 * sizeof(float), allocator);

    float* ptr = reinterpret_cast<float*>(storage.data());

    for(int i = 0; i < 100; ++i)
        ptr[i] = static_cast<float>(i);

    for(int i = 0; i < 100; ++i)
        EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(i));
}

TEST(StorageTest, MoveConstructor)
{
    auto& allocator = nova::AllocatorRegistry::cpu();

    nova::Storage first(1024, allocator);

    void* address = first.data();

    nova::Storage second(std::move(first));

    EXPECT_TRUE(first.empty());

    EXPECT_EQ(first.data(), nullptr);

    EXPECT_EQ(second.data(), address);

    EXPECT_EQ(second.bytes(), 1024);
}


TEST(StorageTest, MoveAssignment)
{
    auto& allocator = nova::AllocatorRegistry::cpu();

    nova::Storage first(1024, allocator);

    nova::Storage second(512, allocator);

    void* address = first.data();

    second = std::move(first);

    EXPECT_TRUE(first.empty());

    EXPECT_EQ(first.data(), nullptr);

    EXPECT_EQ(second.data(), address);

    EXPECT_EQ(second.bytes(), 1024);
}

TEST(StorageTest, LargeAllocation)
{
    auto& allocator = nova::AllocatorRegistry::cpu();

    constexpr std::size_t bytes =
        100 * 1024 * 1024;

    nova::Storage storage(bytes, allocator);

    EXPECT_FALSE(storage.empty());

    EXPECT_EQ(storage.bytes(), bytes);

    EXPECT_NE(storage.data(), nullptr);
}


#include <cstdint>

TEST(StorageTest, Alignment)
{
    auto& allocator = nova::AllocatorRegistry::cpu();

    nova::Storage storage(4096, allocator);

    auto address =
        reinterpret_cast<std::uintptr_t>(storage.data());

    EXPECT_EQ(address % 64, 0);
}


TEST(StorageTest, MultipleStorageObjects)
{
    auto& allocator = nova::AllocatorRegistry::cpu();

    std::vector<nova::Storage> objects;

    for(int i = 0; i < 1000; ++i)
    {
        objects.emplace_back(1024, allocator);
    }

    EXPECT_EQ(objects.size(), 1000);

    for(auto& storage : objects)
    {
        EXPECT_FALSE(storage.empty());

        EXPECT_NE(storage.data(), nullptr);
    }
}


#include <thread>

TEST(StorageTest, MultiThreadedStorage)
{
    auto worker = []()
    {
        auto& allocator =
            nova::AllocatorRegistry::cpu();

        for(int i = 0; i < 2000; ++i)
        {
            nova::Storage storage(1024, allocator);

            int* ptr =
                reinterpret_cast<int*>(storage.data());

            ptr[0] = 42;

            EXPECT_EQ(ptr[0], 42);
        }
    };

    std::vector<std::thread> threads;

    for(int i = 0; i < 16; ++i)
        threads.emplace_back(worker);

    for(auto& t : threads)
        t.join();
}

#include <type_traits>

TEST(StorageTest, CopyTraits)
{
    EXPECT_FALSE(
        std::is_copy_constructible_v<nova::Storage>);

    EXPECT_FALSE(
        std::is_copy_assignable_v<nova::Storage>);
}


TEST(StorageTest, MoveTraits)
{
    EXPECT_TRUE(
        std::is_move_constructible_v<nova::Storage>);

    EXPECT_TRUE(
        std::is_move_assignable_v<nova::Storage>);
}


TEST(StorageTest, EmptyAfterMove)
{
    auto& allocator =
        nova::AllocatorRegistry::cpu();

    nova::Storage first(1024, allocator);

    nova::Storage second(std::move(first));

    EXPECT_TRUE(first.empty());

    EXPECT_EQ(first.bytes(), 0);

    EXPECT_EQ(first.data(), nullptr);
}


TEST(StorageTest, ZeroByteAllocation)
{
    auto& allocator = nova::AllocatorRegistry::cpu();

    nova::Storage storage(0, allocator);

    EXPECT_TRUE(storage.empty());

    EXPECT_EQ(storage.bytes(), 0);

    EXPECT_EQ(storage.data(), nullptr);
}


