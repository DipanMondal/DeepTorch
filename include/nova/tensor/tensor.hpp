#pragma once

#include "nova/tensor/tensor_metadata.hpp"
#include "nova/storage/storage.hpp"
#include "nova/tensor/dtype.hpp"
#include "nova/tensor/shape.hpp"
#include "nova/tensor/detail/indexing.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <limits>



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
			
			// Flatten
			[[nodiscard]]
			Tensor flatten(
				std::size_t start_dim = 0, 
				std::size_t end_dim = std::numeric_limits<std::size_t>::max()
			) const;
			
			// Transpose
			[[nodiscard]]
			Tensor transpose(std::size_t dim0, std::size_t dim1) const;
			
			// Inplace Operation
			void reshape_(const Shape& new_shape);
			
			
			/*------------------------
				Indexing Operation
			------------------------
			*/
			
			template<typename T>
			[[nodiscard]]
			T& at(std::initializer_list<std::size_t> indices);
			
			template<typename T>
			[[nodiscard]]
			const T& at(std::initializer_list<std::size_t> indices) const;
	};
}

#include "nova/tensor/detail/tensor_data.inl"
#include "nova/tensor/detail/tensor_at.inl"