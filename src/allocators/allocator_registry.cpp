#include "nova/allocators/allocator_registry.hpp"
#include "nova/allocators/cpu_allocator.hpp"

namespace nova {
	Allocator& AllocatorRegistry::cpu(){
		static CPUAllocator allocator;
		return allocator;
	}
}