#pragma once

#include "allocator.hpp"

namespace nova {
	class AllocatorRegistry {
		public:
			static Allocator& cpu(); 
			
			// Future
			// static Allocator& cuda();
			// static Allocator& pinned();
			// static Allocator& pool();
	};
}