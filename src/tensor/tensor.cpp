#include "nova/tensor/tensor.hpp"

#include <stdexcept>

#include "nova/allocators/allocator_registry.hpp"

namespace nova
{

	// helper function
	std::shared_ptr<Storage> Tensor::allocate_storage( const Shape& shape,
				const DType& dtype,
				const Device& device) 
	{
		Allocator* allocator = nullptr;

		switch (device.type())
		{
			case DeviceType::CPU:
				allocator = &AllocatorRegistry::cpu();
				break;

			default:
				throw std::runtime_error(
					"Unsupported device.");
		}

		// Allocate storage
		const std::size_t bytes = shape.numel() * dtype.size();

		std::shared_ptr<Storage> storage = std::make_shared<Storage>(bytes,*allocator);		
		
		return storage;
	}
	
	// priave helper function for friend functions of tensor
	const std::shared_ptr<Storage>& Tensor::storage() const noexcept {
		return storage_;
	}

	
	// Constructors
	Tensor::Tensor() noexcept
		: storage_(std::make_shared<Storage>()),
		  metadata_()
	{
	}

	Tensor::Tensor( std::shared_ptr<Storage> storage,
		TensorMetadata metadata)
		: storage_(std::move(storage)),
		  metadata_(std::move(metadata))
	{
		if (!storage_)
		{
			throw std::invalid_argument(
				"Tensor storage cannot be nullptr.");
		}
		
		if (storage_->bytes() < metadata_.required_elements() *metadata_.dtype().size())
		{
			throw std::invalid_argument(
				"Storage is too small for tensor metadata.");
		}
	}

	Tensor::Tensor(Shape shape,
		DType dtype,
		Device device,
		Layout layout)
		: storage_(allocate_storage(shape, dtype, device)),
		  metadata_(TensorMetadata::contiguous(
			  std::move(shape),
			  StorageOffset(0),
			  device,
			  dtype,
			  layout))
	{}

	// Metadata
	const Shape& Tensor::shape() const noexcept {
		return metadata_.shape();
	}

	const Strides& Tensor::strides() const noexcept {
		return metadata_.strides();
	}

	const StorageOffset& Tensor::offset() const noexcept {
		return metadata_.offset();
	}

	const Device& Tensor::device() const noexcept {
		return metadata_.device();
	}

	const DType& Tensor::dtype() const noexcept {
		return metadata_.dtype();
	}

	const Layout& Tensor::layout() const noexcept {
		return metadata_.layout();
	}

	
	// Information
	std::size_t	Tensor::rank() const noexcept {
		return metadata_.rank();
	}

	std::size_t	Tensor::numel() const noexcept {
		return metadata_.numel();
	}

	bool Tensor::empty() const noexcept {
		return metadata_.empty();
	}

	bool
	Tensor::is_contiguous() const noexcept {
		return metadata_.is_contiguous();
	}

	
	// Factory Function
	Tensor Tensor::empty(
		Shape shape,
		DType dtype,
		Device device,
		Layout layout)
	{
		
		// Storage	
		auto storage = allocate_storage(shape, dtype, device);
		
		// Metadata
		TensorMetadata metadata =
			TensorMetadata::contiguous(
				std::move(shape),
				StorageOffset(0),
				device,
				dtype,
				layout);

		
		// Tensor
		return Tensor(
			std::move(storage),
			std::move(metadata));
	}

}