#pragma once

#include <cstddef>

namespace nova {
	class StorageOffset final {
		private:
			std::size_t offset_ = 0;
		
		public:
			StorageOffset() noexcept = default;
			
			StorageOffset(const std::size_t offset) noexcept;
			
			// getter function
			std::size_t value() const noexcept;
	};
}