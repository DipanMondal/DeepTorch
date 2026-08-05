#pragma once

#include <stdexcept>

#include "nova/tensor/detail/indexing.hpp"

namespace nova
{

	template<typename T>
	T& Tensor::at(std::initializer_list<std::size_t> indices) {
		if (dtype() != DType::from<T>())
		{
			throw std::runtime_error(
				"Requested type does not match tensor dtype.");
		}

		const std::size_t linear =
			detail::compute_linear_index(
				metadata_,
				indices);

		return data<T>()[linear];
	}

	template<typename T>
	const T& Tensor::at(
		std::initializer_list<std::size_t> indices) const {
		if (dtype() != DType::from<T>())
		{
			throw std::runtime_error(
				"Requested type does not match tensor dtype.");
		}

		const std::size_t linear =
			detail::compute_linear_index(
				metadata_,
				indices);

		return data<T>()[linear];
	}

}