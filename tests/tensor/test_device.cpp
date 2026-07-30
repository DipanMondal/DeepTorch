#include <gtest/gtest.h>

#include <type_traits>

#include "nova/tensor/device.hpp"

using nova::Device;
using nova::DeviceType;

TEST(DeviceTest, DefaultConstructor)
{
    Device device;

    EXPECT_TRUE(device.is_cpu());

    EXPECT_EQ(device.type(), DeviceType::CPU);

    EXPECT_EQ(device.index(), 0u);
}

TEST(DeviceTest, CpuFactory)
{
    Device device = Device::cpu();

    EXPECT_TRUE(device.is_cpu());

    EXPECT_FALSE(device.is_cuda());

    EXPECT_EQ(device.index(), 0u);
}

TEST(DeviceTest, CudaFactoryDefault)
{
    Device device = Device::cuda();

    EXPECT_TRUE(device.is_cuda());

    EXPECT_FALSE(device.is_cpu());

    EXPECT_EQ(device.index(), 0u);
}

TEST(DeviceTest, CudaFactoryCustomIndex)
{
    Device device = Device::cuda(3);

    EXPECT_TRUE(device.is_cuda());

    EXPECT_EQ(device.index(), 3u);
}

TEST(DeviceTest, ExplicitConstructor)
{
    Device device(DeviceType::CUDA, 7);

    EXPECT_EQ(device.type(), DeviceType::CUDA);

    EXPECT_EQ(device.index(), 7u);
}

TEST(DeviceTest, Equality)
{
    Device a = Device::cuda(1);

    Device b = Device::cuda(1);

    EXPECT_EQ(a, b);
}

TEST(DeviceTest, Inequality)
{
    Device a = Device::cuda(0);

    Device b = Device::cuda(1);

    EXPECT_NE(a, b);
}

TEST(DeviceTest, CpuString)
{
    Device device = Device::cpu();

    EXPECT_EQ(device.to_string(), "cpu");
}

TEST(DeviceTest, CudaString)
{
    Device device = Device::cuda(2);

    EXPECT_EQ(device.to_string(), "cuda:2");
}

TEST(DeviceTest, CopyConstruction)
{
    Device a = Device::cuda(4);

    Device b(a);

    EXPECT_EQ(a, b);
}

TEST(DeviceTest, CopyAssignment)
{
    Device a = Device::cuda(5);

    Device b;

    b = a;

    EXPECT_EQ(a, b);
}

TEST(DeviceTest, MoveConstruction)
{
    Device a = Device::cuda(6);

    Device b(std::move(a));

    EXPECT_TRUE(b.is_cuda());

    EXPECT_EQ(b.index(), 6u);
}

TEST(DeviceTest, MoveAssignment)
{
    Device a = Device::cuda(8);

    Device b;

    b = std::move(a);

    EXPECT_TRUE(b.is_cuda());

    EXPECT_EQ(b.index(), 8u);
}

TEST(DeviceTest, TypeTraits)
{
    EXPECT_TRUE(std::is_copy_constructible_v<Device>);
    EXPECT_TRUE(std::is_copy_assignable_v<Device>);
    EXPECT_TRUE(std::is_move_constructible_v<Device>);
    EXPECT_TRUE(std::is_move_assignable_v<Device>);
	EXPECT_TRUE(Device::cuda(0).is_same_type(Device::cuda(3)));
	EXPECT_FALSE(Device::cpu().is_same_type(Device::cuda()));
}