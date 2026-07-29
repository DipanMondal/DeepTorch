#include "nova/tensor/shape.hpp"
#include <numeric>

namespace nova {
	std::size_t Shape::compute_numel_(const std::vector<std::size_t>&dims){
		if(dims.empty())
			return 0;
		
		return std::accumulate(
			dims.begin(),
			dims.end(),
			std::size_t{1},
			std::multiplies<>{});
	}
	
	
	Shape::Shape(std::initializer_list<std::size_t> dims)
	: dims_(dims)
	{
		numel_ = compute_numel_(dims_);
	}
	
	Shape::Shape(std::vector<std::size_t> dims)
	: dims_(std::move(dims))
	{
		numel_ = compute_numel_(dims_);
	}
	
	std::size_t Shape::rank() const noexcept {
		return dims_.size();
	}
	
	std::size_t Shape::numel() const noexcept {
		return numel_;
	}
	
	bool Shape::empty() const noexcept {
		return dims_.empty();
	}
	
	std::size_t Shape::operator[](std::size_t index) const {
		return dims_.at(index);
	}
	
	const std::vector<std::size_t>& Shape::dimensions() const noexcept {
		return dims_;
	}
	
	bool Shape::operator==(const Shape&other) const noexcept {
		if(dims_.size() != other.dims_.size()) return false;
		
		for(int i=0;i<dims_.size();i++){
			if(dims_[i] != other.dims_[i]) return false;
		}
		
		return true;
	}
	
	bool Shape::operator!=(const Shape& other) const noexcept {
		return !(*this == other);
	}
	
	Shape::const_iterator Shape::begin() const noexcept {
		return dims_.begin();
	}

	Shape::const_iterator Shape::end() const noexcept {
		return dims_.end();
	}

	Shape::value_type Shape::front() const {
		return dims_.front();
	}

	Shape::value_type Shape::back() const {
		return dims_.back();
	}
	
	bool Shape::is_scalar() const noexcept {
		return dims_.empty();
	}
}

