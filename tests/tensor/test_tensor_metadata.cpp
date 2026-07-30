#include <gtest/gtest.h>

#include "nova/tensor/shape.hpp"
#include "nova/tensor/storage_offset.hpp"
#include "nova/tensor/strides.hpp"
#include "nova/tensor/tensor_metadata.hpp"

using namespace nova;

TEST(TensorMetadataTest, Constructor)
{
    Shape shape({2,3});

    Strides strides({3,1});

    StorageOffset offset(0);

    TensorMetadata meta(
        shape,
        strides,
        offset);

    EXPECT_EQ(meta.rank(),2);

    EXPECT_EQ(meta.numel(),6);

    EXPECT_FALSE(meta.empty());

    EXPECT_TRUE(meta.is_contiguous());
}

TEST(TensorMetadataTest, ContiguousFactory)
{
    auto meta =
        TensorMetadata::contiguous(
            Shape({2,3,4}));

    EXPECT_EQ(meta.rank(),3);

    EXPECT_EQ(meta.numel(),24);

    EXPECT_TRUE(meta.is_contiguous());

    EXPECT_EQ(meta.strides()[0],12);

    EXPECT_EQ(meta.strides()[1],4);

    EXPECT_EQ(meta.strides()[2],1);
}

TEST(TensorMetadataTest, ShapeGetter)
{
    auto meta =
        TensorMetadata::contiguous(
            Shape({5,6}));

    EXPECT_EQ(meta.shape()[0],5);

    EXPECT_EQ(meta.shape()[1],6);
}

TEST(TensorMetadataTest, StridesGetter)
{
    auto meta =
        TensorMetadata::contiguous(
            Shape({5,6}));

    EXPECT_EQ(meta.strides()[0],6);

    EXPECT_EQ(meta.strides()[1],1);
}

TEST(TensorMetadataTest, OffsetGetter)
{
    StorageOffset offset(7);

    auto meta =
        TensorMetadata(
            Shape({2,2}),
            Strides({2,1}),
            offset);

    EXPECT_EQ(
        meta.offset().value(),
        7);
}

TEST(TensorMetadataTest, EmptyMetadata)
{
    auto meta =
        TensorMetadata::contiguous(
            Shape());

    EXPECT_TRUE(meta.empty());

    EXPECT_EQ(meta.rank(),0);

    EXPECT_EQ(meta.numel(),0);
}

TEST(TensorMetadataTest, NonContiguous)
{
    TensorMetadata meta(
        Shape({2,3}),
        Strides({1,2}),
        StorageOffset());

    EXPECT_FALSE(meta.is_contiguous());
}

TEST(TensorMetadataTest, InvalidRank)
{
    EXPECT_THROW(
        TensorMetadata(
            Shape({2,3}),
            Strides({6}),
            StorageOffset()),
        std::invalid_argument);
}

TEST(TensorMetadataTest, ZeroDimensionTensor)
{
    auto meta =
        TensorMetadata::contiguous(
            Shape({2,0,5}));

    EXPECT_EQ(meta.numel(),0);

    EXPECT_TRUE(meta.is_contiguous());

    EXPECT_EQ(meta.strides()[0],0);

    EXPECT_EQ(meta.strides()[1],5);

    EXPECT_EQ(meta.strides()[2],1);
}

TEST(TensorMetadataTest, CopyConstruction)
{
    auto meta =
        TensorMetadata::contiguous(
            Shape({2,3}));

    TensorMetadata copy(meta);

    EXPECT_EQ(copy.rank(),2);

    EXPECT_TRUE(copy.is_contiguous());
}

TEST(TensorMetadataTest, CopyAssignment)
{
    TensorMetadata a =
        TensorMetadata::contiguous(
            Shape({2,3}));

    TensorMetadata b =
        TensorMetadata::contiguous(
            Shape({4,5}));

    b = a;

    EXPECT_EQ(b.rank(),2);

    EXPECT_TRUE(b.is_contiguous());
}

TEST(TensorMetadataTest, MoveConstruction)
{
    TensorMetadata meta =
        TensorMetadata::contiguous(
            Shape({2,3}));

    TensorMetadata moved(
        std::move(meta));

    EXPECT_EQ(moved.rank(),2);

    EXPECT_TRUE(moved.is_contiguous());
}

TEST(TensorMetadataTest, MoveAssignment)
{
    TensorMetadata a =
        TensorMetadata::contiguous(
            Shape({2,3}));

    TensorMetadata b =
        TensorMetadata::contiguous(
            Shape({5}));

    b = std::move(a);

    EXPECT_EQ(b.rank(),2);

    EXPECT_TRUE(b.is_contiguous());
}