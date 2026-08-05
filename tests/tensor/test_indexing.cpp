#include <gtest/gtest.h>

#include "nova/tensor/detail/indexing.hpp"

using namespace nova;

TEST(IndexingTest, Contiguous2D)
{
    Shape shape({2,3});
    Strides strides({3,1});
    StorageOffset offset(0);

    EXPECT_EQ(
        detail::compute_linear_index(
            shape,
            strides,
            offset,
            {1,2}),
        5);
}

TEST(IndexingTest, Contiguous3D)
{
    Shape shape({2,3,4});
    Strides strides({12,4,1});
    StorageOffset offset(0);

    EXPECT_EQ(
        detail::compute_linear_index(
            shape,
            strides,
            offset,
            {1,2,3}),
        23);
}

TEST(IndexingTest, OffsetApplied)
{
    Shape shape({2,3});
    Strides strides({3,1});
    StorageOffset offset(10);

    EXPECT_EQ(
        detail::compute_linear_index(
            shape,
            strides,
            offset,
            {1,2}),
        15);
}

TEST(IndexingTest, TransposedTensor)
{
    Shape shape({3,2});
    Strides strides({1,3});
    StorageOffset offset(0);

    EXPECT_EQ(
        detail::compute_linear_index(
            shape,
            strides,
            offset,
            {2,1}),
        5);
}

TEST(IndexingTest, WrongRankThrows)
{
    Shape shape({2,3});
    Strides strides({3,1});

    EXPECT_THROW(
        detail::compute_linear_index(
            shape,
            strides,
            StorageOffset(0),
            {1}),
        std::invalid_argument);
}

TEST(IndexingTest, IndexOutOfBoundsThrows)
{
    Shape shape({2,3});
    Strides strides({3,1});

    EXPECT_THROW(
        detail::compute_linear_index(
            shape,
            strides,
            StorageOffset(0),
            {2,1}),
        std::out_of_range);
}

TEST(IndexingTest, ScalarTensor)
{
    Shape shape({});
    Strides strides({});
    StorageOffset offset(7);

    EXPECT_EQ(
        detail::compute_linear_index(
            shape,
            strides,
            offset,
            {}),
        7);
}

TEST(IndexingTest, EmptyTensorThrows)
{
    Shape shape({0});
    Strides strides({1});
    StorageOffset offset(0);

    EXPECT_THROW(
        detail::compute_linear_index(
            shape,
            strides,
            offset,
            {0}),
        std::out_of_range);
}