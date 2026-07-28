#pragma once

#include <stdexcept>

namespace nova {
	
	// base class of our nova library
	class NovaException : public std::runtime_error {
		public:
			// use all the constructors in out parent class (runtime_error) as it is. 
			using std::runtime_error::runtime_error;
	}
	
	class ShapeError : public NovaException {
		public:
			using NovaException::NovaException;
	}
	
	class DeviceError : public NovaException {
		public:
			using NovaException::NovaException;
	}
	
}