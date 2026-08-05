#include "nova/tensor/strides.hpp"
#include <string>
#include <utility>

namespace nova {
	Strides::Strides(std::initializer_list<value_type> strides) : strides_(strides) {}
	
	Strides::Strides(container_type strides) : strides_(std::move(strides)) {}
	
	Strides Strides::contiguous(const Shape &shape) {
		if (shape.empty())
			return {};

		container_type strides(shape.rank());

		value_type stride = 1;

		for (std::size_t i = shape.rank(); i-- > 0;)
		{
			strides[i] = stride;
			stride *= shape[i];
		}

		return Strides(std::move(strides));
	}
	
	std::size_t Strides::rank() const noexcept {
		return strides_.size();
	}

	bool Strides::empty() const noexcept {
		return strides_.empty();
	}

	Strides::value_type Strides::operator[](std::size_t index) const {
		return strides_.at(index);
	}

	const Strides::container_type& Strides::values() const noexcept	{
		return strides_;
	}
	
	Strides::const_iterator Strides::begin() const noexcept {
		return strides_.begin();
	}

	Strides::const_iterator Strides::end() const noexcept {
		return strides_.end();
	}

	Strides::value_type Strides::front() const {
		return strides_.front();
	}

	Strides::value_type Strides::back() const {
		return strides_.back();
	}
	
	bool Strides::operator==(const Strides& other) const noexcept {
		return strides_ == other.strides_;
	}

	bool Strides::operator!=(const Strides& other) const noexcept {
		return !(*this == other);
	}	
	
	void Strides::validate_dimension(std::size_t dim) const {
		if(dim >= rank()) {
			throw std::out_of_range(
				"dim (" + std::to_string(dim) +
				") is out of range for rank " +
				std::to_string(rank()) + '.'
			);
		}
	}
	
	Strides Strides::transpose(std::size_t dim0, std::size_t dim1) const {
		// validation
		validate_dimension(dim0);
		validate_dimension(dim1);
		
		auto dims = strides_;
		std::swap(dims[dim0], dims[dim1]);
		
		return Strides(std::move(dims));
	}

}