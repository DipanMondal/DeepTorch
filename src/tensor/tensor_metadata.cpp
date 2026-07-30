#include "nova/tensor/tensor_metadata.hpp"
#include <stdexcept>


namespace nova {
	// Constructors
	TensorMetadata::TensorMetadata(Shape shape, Strides strides, StorageOffset offset)
		: shape_(std::move(shape)),
		  strides_(std::move(strides)),
		  offset_(offset)
	{
		if (shape_.rank() != strides_.rank())
		{
			throw std::invalid_argument(
				"Shape and strides must have the same rank.");
		}
	}
	
	TensorMetadata TensorMetadata::contiguous(Shape shape, StorageOffset offset)
	{
		return TensorMetadata(std::move(shape), Strides::contiguous(shape),	offset);
	}
	
	
	// getters
	const Shape& TensorMetadata::shape() const noexcept{
		return shape_;
	}
	
	const Strides& TensorMetadata::strides() const noexcept {
		return strides_;
	}
	
	const StorageOffset& TensorMetadata::offset() const noexcept {
		return offset_;
	}
	
	std::size_t TensorMetadata::rank() const noexcept {
		return shape_.rank();
	}
	
	std::size_t TensorMetadata::numel() const noexcept {
		return shape_.numel();
	}
	
	bool TensorMetadata::empty() const noexcept {
		return shape_.empty();
	}
	
	bool TensorMetadata::is_contiguous() const noexcept {
		return strides_ == Strides::contiguous(shape_);
	}
}