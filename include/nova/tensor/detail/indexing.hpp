#pragma once

#include <cstddef>
#include <span>

#include "nova/tensor/shape.hpp"
#include "nova/tensor/storage_offset.hpp"
#include "nova/tensor/strides.hpp"
#include "nova/tensor/tensor_metadata.hpp"

namespace nova::detail {

	[[nodiscard]]
	std::size_t compute_linear_index(
		const Shape& shape,
		const Strides& strides,
		const StorageOffset& offset,
		std::span<const std::size_t> indices);
		
	std::size_t compute_linear_index(
		const Shape& shape,
		const Strides& strides,
		const StorageOffset& offset,
		std::initializer_list<std::size_t> indices);
	
	std::size_t compute_linear_index(
		const TensorMetadata& metadata,
		std::initializer_list<std::size_t> indices);
	
	std::size_t compute_linear_index(
		const TensorMetadata& metadata,
		std::span<const std::size_t> indices);
}