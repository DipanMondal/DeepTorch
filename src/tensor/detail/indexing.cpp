#include "nova/tensor/detail/indexing.hpp"

#include <stdexcept>
#include <string>

namespace nova::detail {

	std::size_t compute_linear_index(
		const Shape& shape,
		const Strides& strides,
		const StorageOffset& offset,
		std::span<const std::size_t> indices)
	{
		// Validate 
		if (indices.size() != shape.rank())
		{
			throw std::invalid_argument(
				"Expected " +
				std::to_string(shape.rank()) +
				" indices, but got " +
				std::to_string(indices.size()) +
				'.');
		}


		// Compute linear index
		std::size_t linear = offset.value();

		for (std::size_t dim = 0; dim < shape.rank(); ++dim)
		{
			if (indices[dim] >= shape[dim])
			{
				throw std::out_of_range(
					"Index " +
					std::to_string(indices[dim]) +
					" is out of bounds for dimension " +
					std::to_string(dim) +
					" (size = " +
					std::to_string(shape[dim]) +
					").");
			}

			linear += indices[dim] * strides[dim];
		}

		return linear;
	}
	
	
	std::size_t compute_linear_index(
		const Shape& shape,
		const Strides& strides,
		const StorageOffset& offset,
		std::initializer_list<std::size_t> indices)
	{
		return compute_linear_index(
			shape,
			strides,
			offset,
			std::span<const std::size_t>(
				indices.begin(),
				indices.size()));
	}
	
	std::size_t compute_linear_index(
		const TensorMetadata& metadata,
		std::initializer_list<std::size_t> indices) 
	{
		return compute_linear_index(
			metadata.shape(),
			metadata.strides(),
			metadata.offset(),
			indices);
	}
	
	std::size_t compute_linear_index(
		const TensorMetadata& metadata,
		std::span<const std::size_t> indices)
	{
		return compute_linear_index(
			metadata.shape(),
			metadata.strides(),
			metadata.offset(),
			indices);
	}
}