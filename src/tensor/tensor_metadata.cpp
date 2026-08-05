#include "nova/tensor/tensor_metadata.hpp"
#include <stdexcept>


namespace nova {
	// Constructors
	TensorMetadata::TensorMetadata(Shape shape, Strides strides, StorageOffset offset, Device device, DType dtype, Layout layout)
		: shape_(std::move(shape)),
		  strides_(std::move(strides)),
		  offset_(offset),
		  device_(device),
		  dtype_(dtype),
		  layout_(layout)
	{
		if (shape_.rank() != strides_.rank())
		{
			throw std::invalid_argument(
				"Shape and strides must have the same rank.");
		}
	}
	
	TensorMetadata TensorMetadata::contiguous(Shape shape, StorageOffset offset, Device device, DType dtype, Layout layout) {
		return TensorMetadata(std::move(shape), Strides::contiguous(shape),	offset, device, dtype, layout);
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
	
	const Device& TensorMetadata::device() const noexcept {
		return device_;
	}

	const DType& TensorMetadata::dtype() const noexcept {
		return dtype_;
	}

	const Layout& TensorMetadata::layout() const noexcept {
		return layout_;
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
	
	std::size_t TensorMetadata::required_elements() const noexcept {
		std::size_t max_index = 0;

		const auto& strides = strides_.values();

		for (std::size_t i = 0; i < shape_.rank(); ++i)
		{
			if (shape_[i] == 0)
			{
				return 0;
			}

			max_index += (shape_[i] - 1) * strides[i];
		}

		return offset_.value() + max_index + 1;
	}
	
	TensorMetadata TensorMetadata::transpose(
		std::size_t dim0,
		std::size_t dim1) const
	{
		return TensorMetadata(
			shape_.transpose(dim0,dim1),
			strides_.transpose(dim0,dim1),
			offset_,
			device_,
			dtype_,
			layout_);
	}
}