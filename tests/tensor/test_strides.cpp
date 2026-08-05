#include <gtest/gtest.h>

#include <vector>

#include "nova/tensor/shape.hpp"
#include "nova/tensor/strides.hpp"

using nova::Shape;
using nova::Strides;

TEST(StridesTest, DefaultConstructor)
{
    Strides s;

    EXPECT_TRUE(s.empty());
    EXPECT_EQ(s.rank(), 0);
}

TEST(StridesTest, InitializerListConstructor)
{
    Strides s({12,4,1});

    EXPECT_FALSE(s.empty());

    EXPECT_EQ(s.rank(), 3);

    EXPECT_EQ(s[0],12);
    EXPECT_EQ(s[1],4);
    EXPECT_EQ(s[2],1);
}

TEST(StridesTest, VectorConstructor)
{
    std::vector<std::size_t> values={6,2,1};

    Strides s(values);

    EXPECT_EQ(s.rank(),3);

    EXPECT_EQ(s[0],6);
    EXPECT_EQ(s[1],2);
    EXPECT_EQ(s[2],1);
}

TEST(StridesTest, Equality)
{
    Strides a({12,4,1});
    Strides b({12,4,1});

    EXPECT_EQ(a,b);
}

TEST(StridesTest, Inequality)
{
    Strides a({12,4,1});
    Strides b({24,4,1});

    EXPECT_NE(a,b);
}

TEST(StridesTest, Iterator)
{
    Strides s({12,4,1});

    std::vector<std::size_t> result;

    for(auto stride : s)
        result.push_back(stride);

    ASSERT_EQ(result.size(),3);

    EXPECT_EQ(result[0],12);
    EXPECT_EQ(result[1],4);
    EXPECT_EQ(result[2],1);
}

TEST(StridesTest, Front)
{
    Strides s({12,4,1});

    EXPECT_EQ(s.front(),12);
}

TEST(StridesTest, Back)
{
    Strides s({12,4,1});

    EXPECT_EQ(s.back(),1);
}

TEST(StridesTest, BoundsChecking)
{
    Strides s({12,4,1});

    EXPECT_THROW(
        s[3],
        std::out_of_range);
}

TEST(StridesTest, Contiguous2D)
{
    Shape shape({2,3});

    Strides s =
        Strides::contiguous(shape);

    EXPECT_EQ(s.rank(),2);

    EXPECT_EQ(s[0],3);
    EXPECT_EQ(s[1],1);
}

TEST(StridesTest, Contiguous3D)
{
    Shape shape({2,3,4});

    Strides s =
        Strides::contiguous(shape);

    EXPECT_EQ(s.rank(),3);

    EXPECT_EQ(s[0],12);
    EXPECT_EQ(s[1],4);
    EXPECT_EQ(s[2],1);
}

TEST(StridesTest, Contiguous1D)
{
    Shape shape({5});

    Strides s =
        Strides::contiguous(shape);

    EXPECT_EQ(s.rank(),1);

    EXPECT_EQ(s[0],1);
}

TEST(StridesTest, EmptyShape)
{
    Shape shape;

    Strides s =
        Strides::contiguous(shape);

    EXPECT_TRUE(s.empty());
}

TEST(StridesTest, ZeroDimension)
{
    Shape shape({2,0,5});

    Strides s =
        Strides::contiguous(shape);

    EXPECT_EQ(s.rank(),3);

    EXPECT_EQ(s[0],0);
    EXPECT_EQ(s[1],5);
    EXPECT_EQ(s[2],1);
}

TEST(StridesTest, CopyConstructor)
{
    Strides a({12,4,1});

    Strides b(a);

    EXPECT_EQ(a,b);
}

TEST(StridesTest, CopyAssignment)
{
    Strides a({12,4,1});

    Strides b;

    b = a;

    EXPECT_EQ(a,b);
}

TEST(StridesTest, MoveConstructor)
{
    Strides a({12,4,1});

    Strides b(std::move(a));

    EXPECT_EQ(b.rank(),3);

    EXPECT_EQ(b[0],12);
    EXPECT_EQ(b[1],4);
    EXPECT_EQ(b[2],1);
}

TEST(StridesTest, MoveAssignment)
{
    Strides a({12,4,1});

    Strides b;

    b = std::move(a);

    EXPECT_EQ(b.rank(),3);

    EXPECT_EQ(b[0],12);
    EXPECT_EQ(b[1],4);
    EXPECT_EQ(b[2],1);
}

TEST(StridesTest, TransposeBasic)
{
    Strides strides({12,4,1});

    Strides transposed =
        strides.transpose(0,2);

    EXPECT_EQ(transposed[0],1);
    EXPECT_EQ(transposed[1],4);
    EXPECT_EQ(transposed[2],12);
}

TEST(StridesTest, TransposeSameDimension)
{
    Strides strides({12,4,1});

    EXPECT_EQ(
        strides.transpose(1,1),
        strides);
}

TEST(StridesTest, TransposeReverseOrder)
{
    Strides strides({12,4,1});

    EXPECT_EQ(
        strides.transpose(0,2),
        strides.transpose(2,0));
}

TEST(StridesTest, TransposeInvalidDimension)
{
    Strides strides({12,4,1});

    EXPECT_THROW(
        strides.transpose(0,5),
        std::out_of_range);
}