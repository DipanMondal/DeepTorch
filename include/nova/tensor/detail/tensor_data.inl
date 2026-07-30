#pragma once

#include <stdexcept>

#include "nova/core/scalar_type_traits.hpp"

namespace nova
{

	template<typename T>
	T* Tensor::data() {
		if (dtype().scalar_type() != ScalarTypeTraits<T>::scalar_type) {
			throw std::runtime_error(
				"Tensor::data<T>() type does not match tensor dtype.");
		}

		auto* raw = storage_->data();

		raw += offset().value() * sizeof(T);

		return reinterpret_cast<T*>(raw);
	}

	template<typename T>
	const T* Tensor::data() const {
		if (dtype().scalar_type() != ScalarTypeTraits<T>::scalar_type) {
			throw std::runtime_error(
				"Tensor::data<T>() type does not match tensor dtype.");
		}

		auto* raw =
			storage_->data();

		raw +=
			offset().value() * sizeof(T);

		return reinterpret_cast<const T*>(raw);
	}

}