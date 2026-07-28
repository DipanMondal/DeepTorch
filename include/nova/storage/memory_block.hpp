#pragma once

#include <cstddef>

namespace nova {
	struct MemoryBlock
	{
		void* ptr = nullptr;
		std::size_t bytes = 0;
	};
}