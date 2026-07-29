#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

#include "nova/tensor/shape.hpp"

namespace nova {
	class Strides final {
		private:
			std::vector<std::size_t> strides_;
			
		public:
			using value_type = std::size_t;
			using container_type = std::vector<value_type>;
			using const_iterator = container_type::const_iterator;
			
			// Constructors
			
			Strides() noexcept = default;
			
			Strides(std::initializer_list<value_type> strides);
			
			explicit Strides(container_type strides);
			
			
			// Member Functions
			[[nodiscard]]
			static Strides contiguous(const Shape &shape);
			
			[[nodiscard]]
			std::size_t rank() const noexcept;

			[[nodiscard]]
			bool empty() const noexcept;

			[[nodiscard]]
			value_type operator[](std::size_t index) const;

			[[nodiscard]]
			const container_type& values() const noexcept;

			[[nodiscard]]
			const_iterator begin() const noexcept;

			[[nodiscard]]
			const_iterator end() const noexcept;

			[[nodiscard]]
			value_type front() const;

			[[nodiscard]]
			value_type back() const;
			
			// operators
			[[nodiscard]]
			bool operator==(const Strides& other) const noexcept;

			[[nodiscard]]
			bool operator!=(const Strides& other) const noexcept;
	};
}