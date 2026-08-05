#include <gtest/gtest.h>

#include "nova/tensor/tensor.hpp"
#include "nova/tensor/shape.hpp"

using namespace nova;

TEST(ShapeTest, FlattenAll)
{
    Shape shape({2, 3, 4});

    Shape flat = shape.flatten();

    EXPECT_EQ(flat.rank(), 1);
    EXPECT_EQ(flat[0], 24);
}

TEST(ShapeTest, FlattenMiddleDimensions)
{
    Shape shape({2, 3, 4, 5});

    Shape flat = shape.flatten(1, 2);

    EXPECT_EQ(flat.rank(), 3);

    EXPECT_EQ(flat[0], 2);
    EXPECT_EQ(flat[1], 12);
    EXPECT_EQ(flat[2], 5);
}

TEST(ShapeTest, FlattenLastDimensions)
{
    Shape shape({2, 3, 4, 5});

    Shape flat = shape.flatten(2, 3);

    ASSERT_EQ(flat.rank(), 3);

    EXPECT_EQ(flat[0], 2);
    EXPECT_EQ(flat[1], 3);
    EXPECT_EQ(flat[2], 20);
}

TEST(ShapeTest, FlattenSingleDimension)
{
    Shape shape({2,3,4});

    Shape flat = shape.flatten(1,1);

    EXPECT_EQ(flat, shape);
}

TEST(ShapeTest, FlattenVector)
{
    Shape shape({10});

    Shape flat = shape.flatten();

    EXPECT_EQ(flat.rank(),1);
    EXPECT_EQ(flat[0],10);
}

TEST(ShapeTest, FlattenScalar)
{
    Shape shape({});

    Shape flat = shape.flatten();

    EXPECT_EQ(flat.rank(), 1);
    EXPECT_EQ(flat[0], 1);
    EXPECT_EQ(flat.numel(), 1);
}

TEST(ShapeTest, FlattenInvalidRange)
{
    Shape shape({2,3,4});

    EXPECT_THROW(
        shape.flatten(2,1),
        std::invalid_argument);
}

TEST(ShapeTest, FlattenOutOfBounds)
{
    Shape shape({2,3,4});

    EXPECT_THROW(
        shape.flatten(5,5),
        std::out_of_range);
}

TEST(ShapeTest, FlattenPreservesNumel)
{
    Shape shape({2,3,4,5});

    Shape flat = shape.flatten(1,2);

    EXPECT_EQ(
        shape.numel(),
        flat.numel());
}

TEST(TensorTest, FlattenChangesShape)
{
    Tensor a(
        Shape({2,3,4}),
        DType::float32());

    Tensor b = a.flatten();

    EXPECT_EQ(a.rank(),3);

    EXPECT_EQ(b.rank(),1);
    EXPECT_EQ(b.shape()[0],24);
}


TEST(TensorTest, FlattenSharesStorage)
{
    Tensor a(
        Shape({2,3}),
        DType::float32());

    float* pa = a.data<float>();

    for(int i=0;i<6;++i)
        pa[i]=static_cast<float>(i);

    Tensor b=a.flatten();

    float* pb=b.data<float>();

    for(int i=0;i<6;++i)
        EXPECT_FLOAT_EQ(pa[i],pb[i]);
}

TEST(TensorTest, FlattenWriteReflectsOriginal)
{
    Tensor a(
        Shape({2,3}),
        DType::float32());

    Tensor b=a.flatten();

    b.data<float>()[4]=100.f;

    EXPECT_FLOAT_EQ(
        a.data<float>()[4],
        100.f);
}

TEST(TensorTest, FlattenPreservesMetadata)
{
    Tensor a(
        Shape({2,3}),
        DType::float32());

    Tensor b=a.flatten();

    EXPECT_EQ(a.device(),b.device());

    EXPECT_EQ(a.dtype(),b.dtype());

    EXPECT_EQ(a.layout(),b.layout());

    EXPECT_EQ(a.offset(),b.offset());
}

TEST(TensorTest, FlattenDoesNotModifyOriginal)
{
    Tensor a(
        Shape({2,3,4}),
        DType::float32());

    Tensor b=a.flatten();

    EXPECT_EQ(a.rank(),3);

    EXPECT_EQ(a.shape()[0],2);
    EXPECT_EQ(a.shape()[1],3);
    EXPECT_EQ(a.shape()[2],4);

    EXPECT_EQ(b.rank(),1);
    EXPECT_EQ(b.shape()[0],24);
}

TEST(TensorTest, FlattenIsView)
{
    Tensor original(
        Shape({2,3,4}),
        DType::float32());

    auto* p = original.data<float>();

    for(int i = 0; i < 24; ++i)
        p[i] = static_cast<float>(i);

    Tensor flat = original.flatten();

    auto* fp = flat.data<float>();

    fp[17] = 999.0f;

    EXPECT_FLOAT_EQ(
        original.data<float>()[17],
        999.0f);
}

