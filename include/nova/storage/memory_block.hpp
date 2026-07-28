#pragma once

#include <cstddef>

namespace nova {
	struct MemoryBlock
	{
		void* data = nullptr;
		std::size_t bytes = 0;
		std::size_t alignment = 64;
	};
}