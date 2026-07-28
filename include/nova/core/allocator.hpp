#pragma once

#include "nova/storage/memory_block.hpp"

namespace nova {
	class Allocator {
		public:
			virtual ~Allocator() = default;
			
			virtual MemoryBlock allocate(std::size_t bytes,std::size_t alignment = 64) = 0;
			
			virtual void deallocate(const MemoryBlock &block) = 0;
	};
}