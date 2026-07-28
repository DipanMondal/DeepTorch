#pragma once

#include <cstdlib>
#include <iostream>

namespace nova {
	[[noreturn]]
	inline void assertion_failed(const char* expression, const char* file, int line) {
		std::cerr << "Assertion failed: " << expression
            << "\nFile: " << file
            << "\nLine: " << line
            << '\n';
			
		std::abort();
	}
}

#define NOVA_ASSERT(expr)                           \
    do                                              \
    {                                               \
        if (!(expr))                                \
        {                                           \
            ::nova::assertion_failed(               \
                #expr,                              \
                __FILE__,                           \
                __LINE__);                          \
        }                                           \
    } while(false)
		