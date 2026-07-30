#pragma once

#include <cstdint>
#include <string>

namespace nova {
	enum class DeviceType {
		CPU,
		CUDA
	};
	
	class Device final{
		private:
			DeviceType type_ = DeviceType::CPU;
			int index_ = 0;
			
		public:
			constexpr Device() noexcept = default;

			constexpr Device(DeviceType type, int index = 0) noexcept
			: type_(type), index_(index) {}

			[[nodiscard]]
			static constexpr Device cpu() noexcept {
				return Device(DeviceType::CPU);
			}

			[[nodiscard]]
			static constexpr Device cuda(int index = 0) noexcept {
				return Device(DeviceType::CUDA, index);
			}

			[[nodiscard]]
			constexpr DeviceType type() const noexcept {
				return type_;
			}

			[[nodiscard]]
			constexpr int index() const noexcept {
				return index_;
			}

			[[nodiscard]]
			constexpr bool is_cpu() const noexcept {
				return type_ == DeviceType::CPU;
			}

			[[nodiscard]]
			constexpr bool is_cuda() const noexcept {
				return type_ == DeviceType::CUDA;
			}

			[[nodiscard]]
			std::string to_string() const;
			
			[[nodiscard]]
			constexpr bool is_same_type(const Device& other) const noexcept {
				return type_ == other.type_;
			}

			friend constexpr bool operator==(const Device&, const Device&) noexcept = default;
	};	
}