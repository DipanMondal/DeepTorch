#pragma once

#include "nova/core/allocator.hpp"

namespace nova {
	class CPUAllocator final : public Allocator {
		public:
			MemoryBlock allocate(std::size_t bytes) override;
			void deallocate(const MemoryBlock&block) override;
			const char*name() override;
	};
}