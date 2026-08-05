#include <gtest/gtest.h>

#include "nova/tensor/tensor.hpp"
#include "nova/tensor/shape.hpp"

using namespace nova;

TEST(TensorFlattenTest, FlattenChangesShape)
{
    Tensor a(
        Shape({2,3,4}),
        DType::float32());

    Tensor b = a.flatten();

    EXPECT_EQ(a.rank(),3);

    EXPECT_EQ(b.rank(),1);
    EXPECT_EQ(b.shape()[0],24);
}


TEST(TensorFlattenTest, FlattenSharesStorage)
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

TEST(TensorFlattenTest, FlattenWriteReflectsOriginal)
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

TEST(TensorFlattenTest, FlattenPreservesMetadata)
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

TEST(TensorFlattenTest, FlattenDoesNotModifyOriginal)
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

TEST(TensorFlattenTest, FlattenIsView)
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

