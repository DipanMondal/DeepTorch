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

TEST(ShapeTest, Iterator)
{
    Shape shape({2,3,4});

    std::vector<std::size_t> values;

    for(auto dim : shape)
        values.push_back(dim);

    ASSERT_EQ(values.size(), 3);

    EXPECT_EQ(values[0], 2);
    EXPECT_EQ(values[1], 3);
    EXPECT_EQ(values[2], 4);
}

TEST(ShapeTest, Front)
{
    Shape shape({5,6,7});

    EXPECT_EQ(shape.front(), 5);
}

TEST(ShapeTest, Back)
{
    Shape shape({5,6,7});

    EXPECT_EQ(shape.back(), 7);
}

TEST(ShapeTest, isScaler){
	Shape shape1;
	Shape shape2({1});
	
	EXPECT_TRUE(shape1.is_scalar());
	EXPECT_FALSE(shape2.is_scalar());
}

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

TEST(ShapeTest, TransposeBasic)
{
    Shape shape({2, 3, 4});

    Shape transposed = shape.transpose(0, 2);

    EXPECT_EQ(transposed.rank(), 3);

    EXPECT_EQ(transposed[0], 4);
    EXPECT_EQ(transposed[1], 3);
    EXPECT_EQ(transposed[2], 2);
}

TEST(ShapeTest, TransposeAdjacentDimensions)
{
    Shape shape({2, 3, 4, 5});

    Shape transposed = shape.transpose(1, 2);

    EXPECT_EQ(transposed.rank(), 4);

    EXPECT_EQ(transposed[0], 2);
    EXPECT_EQ(transposed[1], 4);
    EXPECT_EQ(transposed[2], 3);
    EXPECT_EQ(transposed[3], 5);
}

TEST(ShapeTest, TransposeSameDimension)
{
    Shape shape({2, 3, 4});

    Shape transposed = shape.transpose(1, 1);

    EXPECT_EQ(transposed, shape);
}

TEST(ShapeTest, TransposeReverseOrder)
{
    Shape shape({2,3,4});

    Shape a = shape.transpose(0,2);
    Shape b = shape.transpose(2,0);

    EXPECT_EQ(a,b);
}

TEST(ShapeTest, TransposeInvalidDimension)
{
    Shape shape({2,3,4});

    EXPECT_THROW(
        shape.transpose(0,3),
        std::out_of_range);

    EXPECT_THROW(
        shape.transpose(5,1),
        std::out_of_range);
}

TEST(ShapeTest, TransposePreservesNumel)
{
    Shape shape({2,3,4,5});

    Shape transposed =
        shape.transpose(1,3);

    EXPECT_EQ(
        shape.numel(),
        transposed.numel());
}