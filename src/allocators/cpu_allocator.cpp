#include "nova/allocators/cpu_allocator.hpp"
#include <cstdlib>
#include <new>

namespace nova {
	MemoryBlock CPUAllocator::allocate(std::size_t bytes,std::size_t alignment) {
		MemoryBlock block{};
		if(bytes==0)
			return block;
		
		block.data = ::operator new(
			bytes,
			std::align_val_t(alignment));

		if(!block.data)
			throw std::bad_alloc();
		
		block.bytes = bytes;
		block.alignment = alignment;
		return block;
	}
	
	void CPUAllocator::deallocate(const MemoryBlock &block)
	{
		::operator delete( block.data, std::align_val_t(block.alignment));
	}
	
	const char* CPUAllocator::name() const noexcept {
		return "CPUAllocator";
	}
}