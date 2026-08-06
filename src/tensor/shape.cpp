#include "nova/tensor/shape.hpp"
#include <numeric>
#include <string>
#include <utility>

namespace nova {
	std::size_t Shape::compute_numel_(const std::vector<std::size_t>&dims){
		if(dims.empty())
			return 1;
		
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
	
	Shape Shape::flatten(std::size_t start_dim, std::size_t end_dim) const {
		if(rank() == 0) {
			return Shape({1});
		}
		if (start_dim >= rank()) {
			throw std::out_of_range(
				"Current Tensor rank "+std::to_string(rank())+" but start_dim "+std::to_string(start_dim)
			);
		}

		if (end_dim == std::numeric_limits<std::size_t>::max()) {
			end_dim = this->rank() - 1;
		}else if (end_dim >= rank()) {
			throw std::out_of_range(
				"Current Tensor rank "+std::to_string(rank())+" but end_dim "+std::to_string(end_dim)
			);
		} 
			
		if (start_dim > end_dim) {
			throw std::invalid_argument(
				"start_dim "+std::to_string(start_dim) + 
				"cant be greater than end_dim "+std::to_string(end_dim)
			);
		}
		
		// calculate new shape
		const int n = start_dim + (this->rank()) - end_dim;
		std::vector<std::size_t> dims;
		dims.reserve(n);
		int i = 0;
		int new_idx=0;
		while(i<rank()) {
			if(i<start_dim || i>end_dim) {
				dims.push_back( this->dims_[i] );
				i++;
			}
			else {
				std::size_t res = 1;
				while(i<=end_dim){
					res *= this->dims_[i];
					i++;
				}
				dims.push_back(res);
			}
		}
		
		return Shape(dims);
	}
	
	void Shape::validate_dimension(std::size_t dim) const {
		if(dim >= rank()) {
			throw std::out_of_range(
				"dim (" + std::to_string(dim) +
				") is out of range for rank " +
				std::to_string(rank()) + '.'
			);
		}
	}
	
	Shape Shape::transpose(std::size_t dim0, std::size_t dim1) const {
		// validation
		validate_dimension(dim0);
		validate_dimension(dim1);
		
		std::vector<std::size_t> order(rank());

		std::iota(order.begin(),order.end(),0);

		std::swap(order[dim0], order[dim1]);

		return permute(order);
	}
	
	Shape Shape::permute(std::span<const std::size_t> order) const {
		// Validate size
		if (order.size() != rank())
		{
			throw std::invalid_argument(
				"Permutation size (" +
				std::to_string(order.size()) +
				") must equal tensor rank (" +
				std::to_string(rank()) +
				").");
		}

		// Validate permutation
		std::vector<bool> visited(rank(), false);

		for (std::size_t dim : order)
		{
			if (dim >= rank())
			{
				throw std::out_of_range(
					"Permutation index " +
					std::to_string(dim) +
					" is out of range.");
			}

			if (visited[dim])
			{
				throw std::invalid_argument(
					"Duplicate dimension " +
					std::to_string(dim) +
					" in permutation.");
			}

			visited[dim] = true;
		}

		// Permute dimensions
		std::vector<std::size_t> dims(rank());

		for (std::size_t i = 0; i < rank(); ++i)
		{
			dims[i] = dims_[order[i]];
		}

		return Shape(std::move(dims));
	}
	
	Shape Shape::permute(std::initializer_list<std::size_t> order) const {
		return permute(
			std::span<const std::size_t>(
				order.begin(),
				order.size()));
	}
}

