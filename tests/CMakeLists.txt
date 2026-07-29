add_executable(

    allocator_tests

    allocators/test_cpu_allocator.cpp
    allocators/test_allocator_registry.cpp
	storage/test_storage.cpp
	tensor/test_shape.cpp
	tensor/test_strides.cpp
	tensor/test_storage_offset.cpp
)

target_link_libraries(

    allocator_tests

    PRIVATE

    nova
    GTest::gtest
    GTest::gtest_main

)

include(GoogleTest)

gtest_discover_tests(allocator_tests)