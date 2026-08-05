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

TEST(TensorTest, AtRead)
{
    Tensor tensor(
        Shape({2,3}),
        DType::float32());

    auto* ptr = tensor.data<float>();

    for (int i = 0; i < 6; ++i)
        ptr[i] = static_cast<float>(i);

    EXPECT_FLOAT_EQ(
        tensor.at<float>({1,2}),
        5.0f);

    EXPECT_FLOAT_EQ(
        tensor.at<float>({0,1}),
        1.0f);
}

TEST(TensorTest, AtWrite)
{
    Tensor tensor(
        Shape({2,3}),
        DType::float32());

    tensor.at<float>({1,2}) = 42.5f;

    EXPECT_FLOAT_EQ(
        tensor.data<float>()[5],
        42.5f);
}

TEST(TensorTest, AtModify)
{
    Tensor tensor(
        Shape({2,3}),
        DType::float32());

    tensor.at<float>({0,1}) = 10.f;

    tensor.at<float>({0,1}) += 5.f;

    EXPECT_FLOAT_EQ(
        tensor.at<float>({0,1}),
        15.f);
}

TEST(TensorTest, AtConst)
{
    Tensor tensor(
        Shape({2,3}),
        DType::float32());

    tensor.at<float>({1,1}) = 99.f;

    const Tensor& ct = tensor;

    EXPECT_FLOAT_EQ(
        ct.at<float>({1,1}),
        99.f);
}

TEST(TensorTest, AtWrongDTypeThrows)
{
    Tensor tensor(
        Shape({2,3}),
        DType::float32());

    EXPECT_THROW(
        tensor.at<int>({0,0}),
        std::runtime_error);
}

TEST(TensorTest, AtWrongRankThrows)
{
    Tensor tensor(
        Shape({2,3}),
        DType::float32());

    EXPECT_THROW(
        tensor.at<float>({1}),
        std::invalid_argument);
}

TEST(TensorTest, AtOutOfBoundsThrows)
{
    Tensor tensor(
        Shape({2,3}),
        DType::float32());

    EXPECT_THROW(
        tensor.at<float>({2,1}),
        std::out_of_range);

    EXPECT_THROW(
        tensor.at<float>({1,3}),
        std::out_of_range);
}

TEST(TensorTest, AtScalar)
{
    Tensor tensor(
        Shape({}),
        DType::float32());

    tensor.at<float>({}) = 3.14f;

    EXPECT_FLOAT_EQ(
        tensor.at<float>({}),
        3.14f);
}

TEST(TensorTest, AtEmptyTensorThrows)
{
    Tensor tensor(
        Shape({0}),
        DType::float32());

    EXPECT_THROW(
        tensor.at<float>({0}),
        std::out_of_range);
}

TEST(TensorTest, AtSharedStorage)
{
    Tensor a(
        Shape({2,3}),
        DType::float32());

    Tensor b = a;

    a.at<float>({1,2}) = 100.f;

    EXPECT_FLOAT_EQ(
        b.at<float>({1,2}),
        100.f);
}

TEST(TensorTest, AtTransposedTensor)
{
    Tensor tensor(
        Shape({2,3}),
        DType::float32());

    auto* ptr = tensor.data<float>();

    for (int i = 0; i < 6; ++i)
        ptr[i] = static_cast<float>(i);

    Tensor transposed =
        tensor.transpose(0,1);

    EXPECT_FLOAT_EQ(
        transposed.at<float>({2,1}),
        5.f);

    EXPECT_FLOAT_EQ(
        transposed.at<float>({0,1}),
        3.f);
	
	for(std::size_t i=0;i<2;i++){
		for(std::size_t j=0;j<3;j++)
			EXPECT_FLOAT_EQ(
				transposed.at<float>({j,i}),
				tensor.at<float>({i,j}));
	}
}


TEST(TensorTest, AtWriteThroughTranspose)
{
    Tensor tensor(
        Shape({2,3}),
        DType::float32());

    Tensor transposed =
        tensor.transpose(0,1);

    transposed.at<float>({2,1}) = 123.f;

    EXPECT_FLOAT_EQ(
        tensor.at<float>({1,2}),
        123.f);
}

TEST(TensorTest, AtAfterReshape)
{
    Tensor tensor(
        Shape({2,3}),
        DType::float32());

    auto* ptr = tensor.data<float>();
    for (int i = 0; i < 6; ++i)
        ptr[i] = static_cast<float>(i);

    Tensor reshaped = tensor.reshape(Shape({3,2}));

    EXPECT_FLOAT_EQ(
        reshaped.at<float>({2,1}),
        5.f);

    reshaped.at<float>({1,0}) = 42.f;

    EXPECT_FLOAT_EQ(
        tensor.at<float>({0,2}),
        42.f);
}

