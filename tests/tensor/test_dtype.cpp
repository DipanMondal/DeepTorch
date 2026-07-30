#include <gtest/gtest.h>

#include <type_traits>

#include "nova/tensor/dtype.hpp"

using namespace nova;

TEST(DTypeTest, DefaultConstructor)
{
    DType dtype;

    EXPECT_EQ(dtype, DType::float32());
    EXPECT_EQ(dtype.scalar_type(), ScalarType::Float32);
}

TEST(DTypeTest, ExplicitConstructor)
{
    DType dtype(ScalarType::Int64);

    EXPECT_EQ(dtype.scalar_type(), ScalarType::Int64);
}

TEST(DTypeTest, BoolFactory)
{
    EXPECT_EQ(DType::bool_().scalar_type(), ScalarType::Bool);
}

TEST(DTypeTest, UInt8Factory)
{
    EXPECT_EQ(DType::uint8().scalar_type(), ScalarType::UInt8);
}

TEST(DTypeTest, Int8Factory)
{
    EXPECT_EQ(DType::int8().scalar_type(), ScalarType::Int8);
}

TEST(DTypeTest, UInt16Factory)
{
    EXPECT_EQ(DType::uint16().scalar_type(), ScalarType::UInt16);
}

TEST(DTypeTest, Int16Factory)
{
    EXPECT_EQ(DType::int16().scalar_type(), ScalarType::Int16);
}

TEST(DTypeTest, UInt32Factory)
{
    EXPECT_EQ(DType::uint32().scalar_type(), ScalarType::UInt32);
}

TEST(DTypeTest, Int32Factory)
{
    EXPECT_EQ(DType::int32().scalar_type(), ScalarType::Int32);
}

TEST(DTypeTest, UInt64Factory)
{
    EXPECT_EQ(DType::uint64().scalar_type(), ScalarType::UInt64);
}

TEST(DTypeTest, Int64Factory)
{
    EXPECT_EQ(DType::int64().scalar_type(), ScalarType::Int64);
}

TEST(DTypeTest, Float16Factory)
{
    EXPECT_EQ(DType::float16().scalar_type(), ScalarType::Float16);
}

TEST(DTypeTest, Float32Factory)
{
    EXPECT_EQ(DType::float32().scalar_type(), ScalarType::Float32);
}

TEST(DTypeTest, Float64Factory)
{
    EXPECT_EQ(DType::float64().scalar_type(), ScalarType::Float64);
}

TEST(DTypeTest, Complex64Factory)
{
    EXPECT_EQ(DType::complex64().scalar_type(), ScalarType::Complex64);
}

TEST(DTypeTest, Complex128Factory)
{
    EXPECT_EQ(DType::complex128().scalar_type(), ScalarType::Complex128);
}

TEST(DTypeTest, Size)
{
    EXPECT_EQ(DType::bool_().size(), 1);

    EXPECT_EQ(DType::uint8().size(), 1);
    EXPECT_EQ(DType::int8().size(), 1);

    EXPECT_EQ(DType::uint16().size(), 2);
    EXPECT_EQ(DType::int16().size(), 2);
    EXPECT_EQ(DType::float16().size(), 2);

    EXPECT_EQ(DType::uint32().size(), 4);
    EXPECT_EQ(DType::int32().size(), 4);
    EXPECT_EQ(DType::float32().size(), 4);

    EXPECT_EQ(DType::uint64().size(), 8);
    EXPECT_EQ(DType::int64().size(), 8);
    EXPECT_EQ(DType::float64().size(), 8);
    EXPECT_EQ(DType::complex64().size(), 8);

    EXPECT_EQ(DType::complex128().size(), 16);
}

TEST(DTypeTest, Alignment)
{
    EXPECT_EQ(DType::bool_().alignment(), 1);

    EXPECT_EQ(DType::int16().alignment(), 2);

    EXPECT_EQ(DType::float32().alignment(), 4);

    EXPECT_EQ(DType::float64().alignment(), 8);

    EXPECT_EQ(DType::complex128().alignment(), 16);
}

TEST(DTypeTest, IsBoolean)
{
    EXPECT_TRUE(DType::bool_().is_boolean());

    EXPECT_FALSE(DType::float32().is_boolean());
}

TEST(DTypeTest, IsIntegral)
{
    EXPECT_TRUE(DType::int32().is_integral());

    EXPECT_TRUE(DType::uint64().is_integral());

    EXPECT_FALSE(DType::float32().is_integral());

    EXPECT_FALSE(DType::complex64().is_integral());
}

TEST(DTypeTest, IsUnsigned)
{
    EXPECT_TRUE(DType::uint8().is_unsigned());

    EXPECT_TRUE(DType::uint64().is_unsigned());

    EXPECT_FALSE(DType::int32().is_unsigned());

    EXPECT_FALSE(DType::float32().is_unsigned());
}

TEST(DTypeTest, IsSigned)
{
    EXPECT_TRUE(DType::int8().is_signed());

    EXPECT_TRUE(DType::int64().is_signed());

    EXPECT_FALSE(DType::uint8().is_signed());

    EXPECT_FALSE(DType::float64().is_signed());
}

TEST(DTypeTest, IsFloating)
{
    EXPECT_TRUE(DType::float16().is_floating());

    EXPECT_TRUE(DType::float32().is_floating());

    EXPECT_TRUE(DType::float64().is_floating());

    EXPECT_FALSE(DType::int32().is_floating());

    EXPECT_FALSE(DType::complex64().is_floating());
}

TEST(DTypeTest, IsComplex)
{
    EXPECT_TRUE(DType::complex64().is_complex());

    EXPECT_TRUE(DType::complex128().is_complex());

    EXPECT_FALSE(DType::float64().is_complex());

    EXPECT_FALSE(DType::int32().is_complex());
}

TEST(DTypeTest, Name)
{
    EXPECT_EQ(DType::bool_().name(), "bool");

    EXPECT_EQ(DType::int32().name(), "int32");

    EXPECT_EQ(DType::float32().name(), "float32");

    EXPECT_EQ(DType::complex128().name(), "complex128");
}

TEST(DTypeTest, ToString)
{
    EXPECT_EQ(DType::bool_().to_string(), "bool");

    EXPECT_EQ(DType::int64().to_string(), "int64");

    EXPECT_EQ(DType::float64().to_string(), "float64");

    EXPECT_EQ(DType::complex64().to_string(), "complex64");
}

TEST(DTypeTest, Equality)
{
    EXPECT_EQ(DType::float32(), DType::float32());

    EXPECT_NE(DType::float32(), DType::float64());

    EXPECT_NE(DType::int32(), DType::uint32());
}

TEST(DTypeTest, CopyConstruction)
{
    DType a = DType::complex128();

    DType b(a);

    EXPECT_EQ(a, b);
}

TEST(DTypeTest, CopyAssignment)
{
    DType a = DType::float16();

    DType b;

    b = a;

    EXPECT_EQ(a, b);
}

TEST(DTypeTest, MoveConstruction)
{
    DType a = DType::int16();

    DType b(std::move(a));

    EXPECT_EQ(b, DType::int16());
}

TEST(DTypeTest, MoveAssignment)
{
    DType a = DType::uint16();

    DType b;

    b = std::move(a);

    EXPECT_EQ(b, DType::uint16());
}

TEST(DTypeTest, TypeTraits)
{
    EXPECT_TRUE(std::is_copy_constructible_v<DType>);
    EXPECT_TRUE(std::is_copy_assignable_v<DType>);
    EXPECT_TRUE(std::is_move_constructible_v<DType>);
    EXPECT_TRUE(std::is_move_assignable_v<DType>);
}