#include <gtest/gtest.h>

#include <type_traits>

#include "nova/tensor/layout.hpp"

using namespace nova;

TEST(LayoutTest, DefaultConstructor)
{
    Layout layout;

    EXPECT_TRUE(layout.is_strided());

    EXPECT_EQ(layout.type(), LayoutType::Strided);
}

TEST(LayoutTest, ExplicitConstructor)
{
    Layout layout(LayoutType::Strided);

    EXPECT_TRUE(layout.is_strided());
}

TEST(LayoutTest, Factory)
{
    Layout layout = Layout::strided();

    EXPECT_TRUE(layout.is_strided());

    EXPECT_EQ(layout.type(), LayoutType::Strided);
}

TEST(LayoutTest, Name)
{
    EXPECT_EQ(Layout::strided().name(), "strided");
}

TEST(LayoutTest, ToString)
{
    EXPECT_EQ(Layout::strided().to_string(), "strided");
}

TEST(LayoutTest, Equality)
{
    EXPECT_EQ(Layout::strided(), Layout::strided());
}

TEST(LayoutTest, CopyConstruction)
{
    Layout a = Layout::strided();

    Layout b(a);

    EXPECT_EQ(a, b);
}

TEST(LayoutTest, CopyAssignment)
{
    Layout a = Layout::strided();

    Layout b;

    b = a;

    EXPECT_EQ(a, b);
}

TEST(LayoutTest, MoveConstruction)
{
    Layout a = Layout::strided();

    Layout b(std::move(a));

    EXPECT_EQ(a, b);
}

TEST(LayoutTest, MoveAssignment)
{
    Layout a = Layout::strided();

    Layout b;

    b = std::move(a);

    EXPECT_EQ(b, Layout::strided());
}

TEST(LayoutTest, TypeTraits)
{
    EXPECT_TRUE(std::is_copy_constructible_v<Layout>);
    EXPECT_TRUE(std::is_copy_assignable_v<Layout>);
    EXPECT_TRUE(std::is_move_constructible_v<Layout>);
    EXPECT_TRUE(std::is_move_assignable_v<Layout>);
}