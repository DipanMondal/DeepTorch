#pragma once

#include <complex>
#include <cstdint>

#include "nova/tensor/dtype.hpp"

namespace nova
{

	template<typename T>
	struct ScalarTypeTraits;

	//--------------------------------------------------
	// Boolean
	//--------------------------------------------------

	template<>
	struct ScalarTypeTraits<bool>
	{
		static constexpr ScalarType scalar_type =
			ScalarType::Bool;
	};

	//--------------------------------------------------
	// Integer
	//--------------------------------------------------

	template<>
	struct ScalarTypeTraits<std::uint8_t>
	{
		static constexpr ScalarType scalar_type =
			ScalarType::UInt8;
	};

	template<>
	struct ScalarTypeTraits<std::int8_t>
	{
		static constexpr ScalarType scalar_type =
			ScalarType::Int8;
	};

	template<>
	struct ScalarTypeTraits<std::uint16_t>
	{
		static constexpr ScalarType scalar_type =
			ScalarType::UInt16;
	};

	template<>
	struct ScalarTypeTraits<std::int16_t>
	{
		static constexpr ScalarType scalar_type =
			ScalarType::Int16;
	};

	template<>
	struct ScalarTypeTraits<std::uint32_t>
	{
		static constexpr ScalarType scalar_type =
			ScalarType::UInt32;
	};

	template<>
	struct ScalarTypeTraits<std::int32_t>
	{
		static constexpr ScalarType scalar_type =
			ScalarType::Int32;
	};

	template<>
	struct ScalarTypeTraits<std::uint64_t>
	{
		static constexpr ScalarType scalar_type =
			ScalarType::UInt64;
	};

	template<>
	struct ScalarTypeTraits<std::int64_t>
	{
		static constexpr ScalarType scalar_type =
			ScalarType::Int64;
	};

	//--------------------------------------------------
	// Floating
	//--------------------------------------------------

	template<>
	struct ScalarTypeTraits<float>
	{
		static constexpr ScalarType scalar_type =
			ScalarType::Float32;
	};

	template<>
	struct ScalarTypeTraits<double>
	{
		static constexpr ScalarType scalar_type =
			ScalarType::Float64;
	};

	//--------------------------------------------------
	// Complex
	//--------------------------------------------------

	template<>
	struct ScalarTypeTraits<std::complex<float>>
	{
		static constexpr ScalarType scalar_type =
			ScalarType::Complex64;
	};

	template<>
	struct ScalarTypeTraits<std::complex<double>>
	{
		static constexpr ScalarType scalar_type =
			ScalarType::Complex128;
	};

}