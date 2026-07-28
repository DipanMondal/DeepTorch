#include "nova/storage/cpu_allocator.hpp"
#include <cstdlib>
#include <new>

namespace nova {
	MemoryBlock CPUAllocator::allocate(std::size_t bytes,std::size_t alignment = 64) {
		MemoryBlock block{};
		if(bytes==0)
			return block;
		
		block.ptr = std::aligned_alloc(
			alignment,
			((bytes+alignment-1)/alignment)*alignment);

		if(!block.ptr)
			throw std::bad_alloc();
		
		block.bytes = bytes;
		return block;
	}
	
	void CPUAllocator::deallocate(const MemoryBlock &block)
	{
		std::free(block.ptr);
	}
	
	const char* CPUAllocator::name() {
		return "CPUAllocator";
	}
}