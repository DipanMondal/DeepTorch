#pragma once

#include <vector>
#include <iostream>
#include <cstddef>
#include <initializer_list>
#include <cstddef>
#include <limits>

namespace nova {
	class Shape final{
		private:
			std::vector<std::size_t> dims_;
			std::size_t numel_=0;
			
			std::size_t compute_numel_(const std::vector<std::size_t>&dims);
		public:
			using value_type = std::size_t;
			using container_type = std::vector<value_type>;
			using const_iterator = container_type::const_iterator;
		
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
			
			[[nodiscard]]
			const_iterator begin() const noexcept;

			[[nodiscard]]
			const_iterator end() const noexcept;

			[[nodiscard]]
			value_type front() const;

			[[nodiscard]]
			value_type back() const;
			
			[[nodiscard]]
			bool is_scalar() const noexcept;
			
			[[nodiscard]]
			Shape flatten(
				std::size_t start_dim = 0, 
				std::size_t end_dim = std::numeric_limits<std::size_t>::max()
			) const;
	};
}