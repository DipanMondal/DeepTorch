#pragma once

#include <string_view>
#include <string>

namespace nova {

	enum class LayoutType {
		Strided
	};

	class Layout final {
		private:
			LayoutType type_;
		
		public:
			constexpr Layout() noexcept : type_(LayoutType::Strided) {}

			constexpr explicit Layout(LayoutType type) noexcept	: type_(type) {}

			[[nodiscard]]
			static constexpr Layout strided() noexcept {
				return Layout(LayoutType::Strided);
			}

			[[nodiscard]]
			constexpr LayoutType type() const noexcept {
				return type_;
			}

			[[nodiscard]]
			constexpr bool is_strided() const noexcept {
				return type_ == LayoutType::Strided;
			}

			[[nodiscard]]
			constexpr std::string_view name() const noexcept {
				return "strided";
			}

			[[nodiscard]]
			std::string to_string() const {
				return std::string(name());
			}

			friend constexpr bool operator==(const Layout&, const Layout&) noexcept = default;
	};

}