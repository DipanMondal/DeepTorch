#include <gtest/gtest.h>

#include "nova/tensor/tensor.hpp"
#include "nova/tensor/shape.hpp"

using namespace nova;

TEST(TensorTransposeTest, TransposeShape)
{
    Tensor tensor(
        Shape({2,3}),
        DType::float32());

    Tensor t =
        tensor.transpose(0,1);

    EXPECT_EQ(t.shape()[0],3);
    EXPECT_EQ(t.shape()[1],2);
}

TEST(TensorTransposeTest, TransposeStrides)
{
    Tensor tensor(
        Shape({2,3}),
        DType::float32());

    Tensor t =
        tensor.transpose(0,1);

    EXPECT_EQ(t.strides()[0],1);
    EXPECT_EQ(t.strides()[1],3);
}

TEST(TensorTransposeTest, TransposeSharesStorage)
{
    Tensor tensor(
        Shape({2,3}),
        DType::float32());

    tensor.data<float>()[4] = 42.f;

    Tensor t =
        tensor.transpose(0,1);

    EXPECT_FLOAT_EQ(
        t.data<float>()[4],
        42.f);
}

TEST(TensorTransposeTest, TransposePreservesMetadata)
{
    Tensor tensor(
        Shape({2,3}),
        DType::float32());

    Tensor t =
        tensor.transpose(0,1);

    EXPECT_EQ(
        tensor.device(),
        t.device());

    EXPECT_EQ(
        tensor.dtype(),
        t.dtype());

    EXPECT_EQ(
        tensor.layout(),
        t.layout());

    EXPECT_EQ(
        tensor.offset(),
        t.offset());
}

TEST(TensorTransposeTest, TransposeDoesNotModifyOriginal)
{
    Tensor tensor(
        Shape({2,3}),
        DType::float32());

    Tensor t =
        tensor.transpose(0,1);

    EXPECT_EQ(
        tensor.shape(),
        Shape({2,3}));

    EXPECT_EQ(
        t.shape(),
        Shape({3,2}));
}

TEST(TensorTransposeTest, TransposeIsNotContiguous)
{
    Tensor tensor(
        Shape({2,3}),
        DType::float32());

    Tensor t =
        tensor.transpose(0,1);

    EXPECT_FALSE(t.is_contiguous());
}
