#pragma once

#include "allocator.hpp"

namespace nova {
	class CPUAllocator final : public Allocator {
		public:
			MemoryBlock allocate(std::size_t bytes,std::size_t alignment = 64) override;
			void deallocate(const MemoryBlock &block) override;
			const char*name() const noexcept override;
	};
}