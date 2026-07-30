#include "nova/tensor/device.hpp"

namespace nova {
	std::string Device::to_string() const
	{
		switch (type_)
		{
			case DeviceType::CPU:
				return "cpu";

			case DeviceType::CUDA:
				return "cuda:" + std::to_string(index_);
		}

		return "unknown";
	}
	
	std::ostream& operator<<(std::ostream& os, const Device& device) {
		return os << device.to_string();
	}

}