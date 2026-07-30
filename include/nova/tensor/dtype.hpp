#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace nova {

	enum class ScalarType {
		Bool,

		UInt8,
		Int8,

		UInt16,
		Int16,

		UInt32,
		Int32,

		UInt64,
		Int64,

		Float16,
		Float32,
		Float64,

		Complex64,
		Complex128
	};

	class DType final {
		private:
			ScalarType scalar_;
		
		public:
			constexpr DType() noexcept : scalar_(ScalarType::Float32) {}

			constexpr explicit DType(ScalarType scalar) noexcept : scalar_(scalar) {}

			// Factory functions

			static constexpr DType bool_() noexcept { return DType(ScalarType::Bool); }

			static constexpr DType uint8() noexcept { return DType(ScalarType::UInt8); }

			static constexpr DType int8() noexcept { return DType(ScalarType::Int8); }

			static constexpr DType uint16() noexcept { return DType(ScalarType::UInt16); }

			static constexpr DType int16() noexcept { return DType(ScalarType::Int16); }

			static constexpr DType uint32() noexcept { return DType(ScalarType::UInt32); }

			static constexpr DType int32() noexcept { return DType(ScalarType::Int32); }

			static constexpr DType uint64() noexcept { return DType(ScalarType::UInt64); }

			static constexpr DType int64() noexcept { return DType(ScalarType::Int64); }

			static constexpr DType float16() noexcept { return DType(ScalarType::Float16); }

			static constexpr DType float32() noexcept { return DType(ScalarType::Float32); }

			static constexpr DType float64() noexcept { return DType(ScalarType::Float64); }

			static constexpr DType complex64() noexcept { return DType(ScalarType::Complex64); }

			static constexpr DType complex128() noexcept { return DType(ScalarType::Complex128); }

			[[nodiscard]]
			constexpr ScalarType scalar_type() const noexcept {
				return scalar_;
			}

			[[nodiscard]]
			constexpr std::size_t size() const noexcept {
				switch (scalar_) {
					case ScalarType::Bool: return 1;

					case ScalarType::UInt8:
					case ScalarType::Int8:
						return 1;

					case ScalarType::UInt16:
					case ScalarType::Int16:
					case ScalarType::Float16:
						return 2;

					case ScalarType::UInt32:
					case ScalarType::Int32:
					case ScalarType::Float32:
						return 4;

					case ScalarType::UInt64:
					case ScalarType::Int64:
					case ScalarType::Float64:
					case ScalarType::Complex64:
						return 8;

					case ScalarType::Complex128:
						return 16;
				}

				return 0;
			}

			[[nodiscard]]
			constexpr std::size_t alignment() const noexcept {
				return size();
			}

			[[nodiscard]]
			constexpr bool is_boolean() const noexcept {
				return scalar_ == ScalarType::Bool;
			}

			[[nodiscard]]
			constexpr bool is_integral() const noexcept {
				switch (scalar_) {
					case ScalarType::UInt8:
					case ScalarType::UInt16:
					case ScalarType::UInt32:
					case ScalarType::UInt64:

					case ScalarType::Int8:
					case ScalarType::Int16:
					case ScalarType::Int32:
					case ScalarType::Int64:

						return true;

					default:
						return false;
				}
			}

			[[nodiscard]]
			constexpr bool is_unsigned() const noexcept {
				switch (scalar_) {
					case ScalarType::UInt8:
					case ScalarType::UInt16:
					case ScalarType::UInt32:
					case ScalarType::UInt64:
						return true;

					default:
						return false;
				}
			}

			[[nodiscard]]
			constexpr bool is_signed() const noexcept {
				switch (scalar_) {
					case ScalarType::Int8:
					case ScalarType::Int16:
					case ScalarType::Int32:
					case ScalarType::Int64:
						return true;

					default:
						return false;
				}
			}

			[[nodiscard]]
			constexpr bool is_floating() const noexcept {
				switch (scalar_) {
					case ScalarType::Float16:
					case ScalarType::Float32:
					case ScalarType::Float64:
						return true;

					default:
						return false;
				}
			}

			[[nodiscard]]
			constexpr bool is_complex() const noexcept {
				return scalar_ == ScalarType::Complex64 || scalar_ == ScalarType::Complex128;
			}

			[[nodiscard]]
			constexpr std::string_view name() const noexcept {
				switch (scalar_) {
					case ScalarType::Bool: return "bool";

					case ScalarType::UInt8: return "uint8";
					case ScalarType::Int8: return "int8";

					case ScalarType::UInt16: return "uint16";
					case ScalarType::Int16: return "int16";

					case ScalarType::UInt32: return "uint32";
					case ScalarType::Int32: return "int32";

					case ScalarType::UInt64: return "uint64";
					case ScalarType::Int64: return "int64";

					case ScalarType::Float16: return "float16";
					case ScalarType::Float32: return "float32";
					case ScalarType::Float64: return "float64";

					case ScalarType::Complex64: return "complex64";
					case ScalarType::Complex128: return "complex128";
				}

				return "unknown";
			}

			[[nodiscard]]
			std::string to_string() const {
				return std::string(name());
			}

			friend constexpr bool operator==(const DType&, const DType&) noexcept = default;
	};

} 