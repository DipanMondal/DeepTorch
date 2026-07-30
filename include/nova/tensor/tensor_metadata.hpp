#pragma once

#include "nova/tensor/shape.hpp"
#include "nova/tensor/strides.hpp"
#include "nova/tensor/storage_offset.hpp"
#include "nova/tensor/device.hpp"
#include "nova/tensor/dtype.hpp"
#include "nova/tensor/layout.hpp"

namespace nova {
	class TensorMetadata final {
		private:
			Shape shape_;
			StorageOffset offset_;
			Strides strides_;
			
			Device device_;
			DType dtype_;
			Layout layout_;
			
		public:
			// constructors
			TensorMetadata() noexcept = default;
			TensorMetadata(Shape shape, Strides strides, StorageOffset offset, Device device = Device::cpu(), DType dtype = DType::float32(), Layout layout = Layout::strided());
			
			// Factory Function
			[[nodiscard]]
			static TensorMetadata contiguous(Shape shape, StorageOffset offset = StorageOffset{}, Device device = Device::cpu(),DType dtype = DType::float32(), Layout layout = Layout::strided());
			
			// getters
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
			
			[[nodiscard]]
			std::size_t rank() const noexcept;

			[[nodiscard]]
			std::size_t numel() const noexcept;

			[[nodiscard]]
			bool empty() const noexcept;

			[[nodiscard]]
			bool is_contiguous() const noexcept;
			
			[[nodiscard]]
			std::size_t required_elements() const noexcept;
			
			friend bool operator==(const TensorMetadata&, const TensorMetadata&) = default;
	};
}