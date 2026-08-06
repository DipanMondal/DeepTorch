#include <gtest/gtest.h>

#include "nova/tensor/tensor.hpp"
#include "nova/tensor/shape.hpp"

using namespace nova;


TEST(TensorTest, PermuteShape)
{
    Tensor t(
        Shape({2,3,4}),
        DType::float32());

    Tensor p = t.permute({2,0,1});

    EXPECT_EQ(
        p.shape(),
        Shape({4,2,3}));
}

TEST(TensorTest, PermuteStrides)
{
    Tensor t(
        Shape({2,3,4}),
        DType::float32());

    Tensor p = t.permute({2,0,1});

    EXPECT_EQ(
        p.strides(),
        Strides({1,12,4}));
}


TEST(TensorTest, PermuteSharesStorage)
{
    Tensor t(
        Shape({2,3,4}),
        DType::float32());

    Tensor p = t.permute({2,0,1});

    EXPECT_EQ(
        t.data<float>(),
        p.data<float>());
}

TEST(TensorTest, PermuteLogicalAccess)
{
    Tensor t(
        Shape({2,3,4}),
        DType::float32());

    float value = 123.0f;

    t.at<float>({1,2,3}) = value;

    Tensor p = t.permute({2,0,1});

    EXPECT_FLOAT_EQ(
        p.at<float>({3,1,2}),
        value);
}

