#include <gtest/gtest.h>
#include <cstdint>

#include "nova/allocators/allocator_registry.hpp"


TEST(AllocatorRegistryTest, Singleton)
{
    auto& a =
        nova::AllocatorRegistry::cpu();

    auto& b =
        nova::AllocatorRegistry::cpu();

    EXPECT_EQ(&a, &b);
}