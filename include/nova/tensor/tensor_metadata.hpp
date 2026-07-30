#pragma once

#include "nova/tensor/shape.hpp"
#include "nova/tensor/strides.hpp"
#include "nova/tensor/storage_offset.hpp"

namespace nova {
	class TensorMetadata final {
		private:
			Shape shape_;
			StorageOffset offset_;
			Strides strides_;
			
		public:
			// constructors
			TensorMetadata() noexcept = default;
			TensorMetadata(Shape shape, Strides strides, StorageOffset offset);
			
			[[nodiscard]]
			static TensorMetadata contiguous(Shape shape, StorageOffset offset = StorageOffset{});
			
			// getters
			[[nodiscard]]
			const Shape& shape() const noexcept;
			
			[[nodiscard]]
			const Strides& strides() const noexcept;
			
			[[nodiscard]]
			const StorageOffset& offset() const noexcept;
			
			[[nodiscard]]
			std::size_t rank() const noexcept;

			[[nodiscard]]
			std::size_t numel() const noexcept;

			[[nodiscard]]
			bool empty() const noexcept;

			[[nodiscard]]
			bool is_contiguous() const noexcept;
	};
}