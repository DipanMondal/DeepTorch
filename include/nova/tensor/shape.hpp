#pragma once

#include <vector>
#include <iostream>

namespace nova {
	class Shape {
		private:
			std::vector<std::size_t> dims_;
			std::size_t numel_=0;
			
			std::size_t compute_numel_(const std::vector<std::size_t>&dims);
		public:
			Shape() noexcept = default;
			
			Shape(std::initializer_list<std::size_t> dims);
			
			explicit Shape(std::vector<std::size_t> dims);
			
			std::size_t rank() const noexcept;
			
			std::size_t numel() const noexcept;
			
			bool empty() const noexcept;
			
			std::size_t operator[](std::size_t index) const;

			const std::vector<std::size_t>& dimensions() const noexcept;

			bool operator==(const Shape&) const noexcept;

			bool operator!=(const Shape&) const noexcept;
	};
}