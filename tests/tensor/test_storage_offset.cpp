#include <gtest/gtest.h>

#include "nova/tensor/storage_offset.hpp"
#include <cstddef>

using nova::StorageOffset;

TEST(StorageOffsetTest, DefaultConstructor)
{
	StorageOffset offset;
	
	EXPECT_EQ(0,offset.value());
}


TEST(StorageOffsetTest, OffsetValue)
{
	std::size_t val = 47;
	StorageOffset offset(val);
	
	EXPECT_EQ(val,offset.value());
}
