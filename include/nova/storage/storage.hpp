#pragma once

#include "nova/allocators/allocator.hpp"
#include "nova/storage/memory_block.hpp"


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
			void* data() noexcept;
			
			const void* data() const noexcept;

			std::size_t bytes() const noexcept;

			bool empty() const noexcept;

			
			~Storage();
	};
}