#pragma once

#include "nova/tensor/tensor_metadata.hpp"
#include "nova/storage/storage.hpp"
#include "nova/tensor/dtype.hpp"
#include <cstddef>
#include <memory>

#include "nova/tensor/shape.hpp"

namespace nova {
	class Tensor final {
		private:
			std::shared_ptr<Storage> storage_;
			TensorMetadata metadata_;
			
			static std::shared_ptr<Storage>	allocate_storage( const Shape& shape,
				const DType& dtype,
				const Device& device);
			
		public:
			// Constructors
			Tensor() noexcept;
			
			Tensor(std::shared_ptr<Storage> storage, TensorMetadata metadata);
			
			explicit Tensor(Shape shape, 
				DType dtype = DType::float32(),
				Device device = Device::cpu(),
				Layout layout = Layout::strided());
			
			// RAII rule of five
			Tensor(const Tensor&) = default;
			Tensor(Tensor&&) = default;
			
			Tensor& operator=(const Tensor&) = default;
			Tensor& operator=(Tensor&&) noexcept = default;
			
			~Tensor() = default;
			
			
			// Metadata
			 [[nodiscard]]
			const Shape& shape() const noexcept;

			[[nodiscard]]
			const Strides& strides() const noexcept;

			[[nodiscard]]
			const StorageOffset& offset() const noexcept;

			[[nodiscard]]
			const Device& device() const noexcept;

			[[nodiscard]]
			const DType& dtype() const noexcept;

			[[nodiscard]]
			const Layout& layout() const noexcept;
			
			
			// Information
			std::size_t rank() const noexcept;

			[[nodiscard]]
			std::size_t numel() const noexcept;

			[[nodiscard]]
			bool empty() const noexcept;

			[[nodiscard]]
			bool is_contiguous() const noexcept;
			
			
			// Raw Memory
			template<typename T>
			[[nodiscard]]
			T* data();
			
			template<typename T>
			[[nodiscard]]
			const T* data() const;
			
			
			// Factory function
			static Tensor empty( Shape shape,
				DType dtype = DType::float32(),
				Device device = Device::cpu(),
				Layout layout = Layout::strided());
				
			
			/*
				View Schemas
			*/
			
			// Reshape
			[[nodiscard]]
			Tensor reshape(const Shape& new_shape) const;
	};
}

#include "nova/tensor/detail/tensor_data.inl"