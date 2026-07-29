#include "nova/tensor/storage_offset.hpp"

namespace nova {
	StorageOffset::StorageOffset(std::size_t offset) noexcept : offset_(offset) {}
	
	//getter function
	std::size_t StorageOffset::value() const noexcept {
		return offset_;
	}
}