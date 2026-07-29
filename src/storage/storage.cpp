#include "nova/storage/storage.hpp"

namespace nova {
	// Constructors
	
	Storage::Storage(std::size_t bytes, Allocator&allocator, std::size_t alignment) : allocator_(&allocator){
		block_ = allocator_->allocate(bytes, alignment);
	}
	
	Storage::Storage(Storage&&other) noexcept 
	: block_(other.block_), allocator_(other.allocator_)
	{
		other.block_ = {};
		other.allocator_ = nullptr;
	}
	
	Storage& Storage::operator=(Storage&&other) noexcept{
		if(this != &other){
			if (allocator_ && block_.data){
				allocator_->deallocate(block_);
			}

			
			block_ = other.block_;
			allocator_ = other.allocator_;
			
			other.block_ = {};
			other.allocator_ = nullptr;
		}
		
		return *this;
	}
	
	
	
	// getters
	void* Storage::data() noexcept {
		return block_.data;
	}

	const void* Storage::data() const noexcept{
		return block_.data;
	}

	std::size_t Storage::bytes() const noexcept{
		return block_.bytes;
	}

	bool Storage::empty() const noexcept{
		return block_.data == nullptr || block_.bytes == 0;
	}
	
	
	// destructor
	Storage::~Storage(){
		if (allocator_ && block_.data)
		{
			allocator_->deallocate(block_);
		}
	}
}