#pragma once

#include "nova/allocators/allocator.hpp"
#include "nova/storage/memory_block.hpp"
#include <cstddef>


namespace nova {
	class Storage{
		private:
			MemoryBlock block_;
			Allocator *allocator_;
		
		public:
			Storage() noexcept = default;
			Storage(std::size_t bytes, Allocator &allocator, std::size_t alignment = 64) ;
			
			Storage(const Storage&) = delete;
			Storage& operator=(const Storage&) = delete;
			
			Storage(Storage&&) noexcept;
			Storage& operator=(Storage&&) noexcept;
			
			// getter
			[[nodiscard]]
			std::byte* data() noexcept;

			[[nodiscard]]
			const std::byte* data() const noexcept;

			[[nodiscard]]
			std::size_t bytes() const noexcept;

			[[nodiscard]]
			bool empty() const noexcept;

			
			~Storage();
	};
}