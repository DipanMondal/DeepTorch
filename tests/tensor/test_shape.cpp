#include <gtest/gtest.h>

#include <type_traits>

#include "nova/tensor/shape.hpp"

using nova::Shape;

TEST(ShapeTest, DefaultConstructor)
{
    Shape shape;

    EXPECT_TRUE(shape.empty());
    EXPECT_EQ(shape.rank(), 0);
    EXPECT_EQ(shape.numel(), 0);
}

TEST(ShapeTest, InitializerListConstructor)
{
    Shape shape({2,3,4});

    EXPECT_FALSE(shape.empty());
    EXPECT_EQ(shape.rank(), 3);
    EXPECT_EQ(shape.numel(), 24);
}

TEST(ShapeTest, VectorConstructor)
{
    std::vector<std::size_t> dims = {5,6};

    Shape shape(dims);

    EXPECT_EQ(shape.rank(), 2);
    EXPECT_EQ(shape.numel(), 30);
}

TEST(ShapeTest, IndexOperator)
{
    Shape shape({10,20,30});

    EXPECT_EQ(shape[0], 10);
    EXPECT_EQ(shape[1], 20);
    EXPECT_EQ(shape[2], 30);
}

TEST(ShapeTest, BoundsChecking)
{
    Shape shape({2,3,4});

    EXPECT_THROW(
        shape[3],
        std::out_of_range);
}

TEST(ShapeTest, Equality)
{
    Shape a({2,3,4});

    Shape b({2,3,4});

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
}

TEST(ShapeTest, Inequality)
{
    Shape a({2,3,4});

    Shape b({2,3,5});

    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a == b);
}

TEST(ShapeTest, ZeroDimensionTensor)
{
    Shape shape({2,0,5});

    EXPECT_EQ(shape.rank(), 3);
    EXPECT_EQ(shape.numel(), 0);
}

TEST(ShapeTest, CopyConstructor)
{
    Shape original({2,3,4});

    Shape copy(original);

    EXPECT_EQ(original, copy);
}

TEST(ShapeTest, CopyAssignment)
{
    Shape original({4,5,6});

    Shape copy;

    copy = original;

    EXPECT_EQ(original, copy);
}

TEST(ShapeTest, MoveConstructor)
{
    Shape original({2,3,4});

    Shape moved(std::move(original));

    EXPECT_EQ(moved.rank(), 3);
    EXPECT_EQ(moved.numel(), 24);
}

TEST(ShapeTest, MoveAssignment)
{
    Shape original({7,8});

    Shape moved;

    moved = std::move(original);

    EXPECT_EQ(moved.rank(), 2);
    EXPECT_EQ(moved.numel(), 56);
}

TEST(ShapeTest, DimensionsGetter)
{
    Shape shape({1,2,3});

    const auto& dims = shape.dimensions();

    ASSERT_EQ(dims.size(), 3);

    EXPECT_EQ(dims[0], 1);
    EXPECT_EQ(dims[1], 2);
    EXPECT_EQ(dims[2], 3);
}

TEST(ShapeTest, LargeShape)
{
    Shape shape({10,20,30,40});

    EXPECT_EQ(shape.rank(), 4);

    EXPECT_EQ(shape.numel(), 240000);
}

TEST(ShapeTest, ValueTypeTraits)
{
    EXPECT_TRUE(std::is_copy_constructible_v<Shape>);
    EXPECT_TRUE(std::is_copy_assignable_v<Shape>);
    EXPECT_TRUE(std::is_move_constructible_v<Shape>);
    EXPECT_TRUE(std::is_move_assignable_v<Shape>);
}