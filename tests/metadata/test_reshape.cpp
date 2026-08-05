#include <gtest/gtest.h>

#include "nova/tensor/tensor.hpp"
#include "nova/tensor/shape.hpp"

using namespace nova;

TEST(ViewTest, ReshapeChangesShape)
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

TEST(ViewTest, ReshapeSharesStorage)
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

TEST(ViewTest, ReshapeWriteReflectsOriginal)
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

TEST(ViewTest, ReshapeDifferentNumelThrows)
{
    Tensor a(
        Shape({2, 3}),
        DType::float32());

    EXPECT_THROW(
        a.reshape(
            Shape({5})),
        std::invalid_argument);
}

TEST(ViewTest, ReshapeCreatesNewMetadata)
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