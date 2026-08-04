#include <gtest/gtest.h>

#include "nova/tensor/tensor.hpp"
#include "nova/allocators/allocator_registry.hpp"

using namespace nova;

TEST(TensorTest, DefaultConstructor)
{
    Tensor tensor;

    EXPECT_TRUE(tensor.empty());
    EXPECT_EQ(tensor.rank(), 0);
    EXPECT_EQ(tensor.numel(), 0);

    EXPECT_EQ(tensor.device(), Device::cpu());
    EXPECT_EQ(tensor.dtype(), DType::float32());
    EXPECT_EQ(tensor.layout(), Layout::strided());
}

TEST(TensorTest, ShapeConstructor)
{
	Tensor tensor( Shape({2, 3}), DType::float32(), Device::cpu(), Layout::strided()); 
	EXPECT_EQ(tensor.rank(), 2); 
	EXPECT_EQ(tensor.numel(), 6); 
	EXPECT_EQ(tensor.shape()[0], 2); 
	EXPECT_EQ(tensor.shape()[1], 3); 
	EXPECT_EQ(tensor.dtype(), DType::float32()); 
	EXPECT_EQ(tensor.device(), Device::cpu());
}


TEST(TensorTest, ContiguousStrides)
{
    Tensor tensor(
        Shape({2, 3, 4}),
        DType::float32());

    EXPECT_EQ(tensor.strides()[0], 12);
    EXPECT_EQ(tensor.strides()[1], 4);
    EXPECT_EQ(tensor.strides()[2], 1);
}

TEST(TensorTest, StorageOffset)
{
    Tensor tensor(
        Shape({5}),
        DType::float32());

    EXPECT_EQ(
        tensor.offset().value(),
        0);
}

TEST(TensorTest, DataReadWriteFloat)
{
    Tensor tensor(
        Shape({100}),
        DType::float32());

    float* ptr = tensor.data<float>();

    for (int i = 0; i < 100; ++i)
        ptr[i] = static_cast<float>(i);

    for (int i = 0; i < 100; ++i)
        EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(i));
}


TEST(TensorTest, ConstData)
{
    Tensor tensor(
        Shape({10}),
        DType::float32());

    const Tensor& ct = tensor;

    EXPECT_NE(
        ct.data<float>(),
        nullptr);
}

TEST(TensorTest, WrongDataTypeThrows)
{
    Tensor tensor(
        Shape({10}),
        DType::float32());

    EXPECT_THROW(
        tensor.data<int>(),
        std::runtime_error);
}


TEST(TensorTest, EmptyFactory)
{
    auto tensor =
        Tensor::empty(
            Shape({4, 5}),
            DType::int32(),
            Device::cpu(),
            Layout::strided());

    EXPECT_EQ(tensor.numel(), 20);

    EXPECT_EQ(
        tensor.dtype(),
        DType::int32());

    EXPECT_EQ(
        tensor.device(),
        Device::cpu());
}


TEST(TensorTest, CopyConstructor)
{
    Tensor a(
        Shape({5}),
        DType::float32());

    Tensor b = a;

    EXPECT_EQ(a.shape(), b.shape());
    EXPECT_EQ(a.dtype(), b.dtype());
}


TEST(TensorTest, CopySharesStorage)
{
    Tensor a(
        Shape({5}),
        DType::float32());

    Tensor b = a;

    a.data<float>()[0] = 123.f;

    EXPECT_FLOAT_EQ(
        b.data<float>()[0],
        123.f);
}


TEST(TensorTest, MoveConstructor)
{
    Tensor a(
        Shape({5}),
        DType::float32());

    Tensor b(std::move(a));

    EXPECT_EQ(
        b.numel(),
        5);
}

TEST(TensorTest, TypeTraits)
{
    EXPECT_TRUE(
        std::is_copy_constructible_v<Tensor>);

    EXPECT_TRUE(
        std::is_move_constructible_v<Tensor>);

    EXPECT_TRUE(
        std::is_copy_assignable_v<Tensor>);

    EXPECT_TRUE(
        std::is_move_assignable_v<Tensor>);
}

TEST(TensorTest, StorageTooSmallThrows)
{
    auto storage = std::make_shared<Storage>(sizeof(float), AllocatorRegistry::cpu()); // Only enough for 1 float

    TensorMetadata metadata = TensorMetadata::contiguous(
        Shape({10}),
        StorageOffset(0),
        Device::cpu(),
        DType::float32(),
        Layout::strided());

    EXPECT_THROW(
        Tensor(storage, metadata),
        std::invalid_argument);
}

TEST(TensorTest, ReshapeChangesShape)
{
    Tensor a(
        Shape({2, 3}),
        DType::float32());

    Tensor b =
        a.reshape(
            Shape({6}));

    EXPECT_EQ(b.rank(), 1);
    EXPECT_EQ(b.shape()[0], 6);

    EXPECT_EQ(a.rank(), 2);
    EXPECT_EQ(a.shape()[0], 2);
    EXPECT_EQ(a.shape()[1], 3);
}

TEST(TensorTest, ReshapeSharesStorage)
{
    Tensor a(
        Shape({2, 3}),
        DType::float32());

    float* pa = a.data<float>();

    for (int i = 0; i < 6; ++i)
        pa[i] = static_cast<float>(i);

    Tensor b =
        a.reshape(
            Shape({6}));

    float* pb = b.data<float>();

    for (int i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(
            pa[i],
            pb[i]);
}

TEST(TensorTest, ReshapeWriteReflectsOriginal)
{
    Tensor a(
        Shape({2, 3}),
        DType::float32());

    Tensor b =
        a.reshape(
            Shape({6}));

    b.data<float>()[4] = 42.0f;

    EXPECT_FLOAT_EQ(
        a.data<float>()[4],
        42.0f);
}

TEST(TensorTest, ReshapeDifferentNumelThrows)
{
    Tensor a(
        Shape({2, 3}),
        DType::float32());

    EXPECT_THROW(
        a.reshape(
            Shape({5})),
        std::invalid_argument);
}

TEST(TensorTest, ReshapeCreatesNewMetadata)
{
    Tensor a(
        Shape({2, 3}),
        DType::float32());

    Tensor b =
        a.reshape(
            Shape({6}));

    EXPECT_NE(
        a.rank(),
        b.rank());

    EXPECT_EQ(
        a.device(),
        b.device());

    EXPECT_EQ(
        a.dtype(),
        b.dtype());

    EXPECT_EQ(
        a.layout(),
        b.layout());
}