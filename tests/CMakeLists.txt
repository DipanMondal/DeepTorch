add_executable(

    allocator_tests

    allocators/test_cpu_allocator.cpp
    allocators/test_allocator_registry.cpp
)

add_executable(

	storage_tests
	
	storage/test_storage.cpp
)

add_executable(
	
	tensor_tests
	
	tensor/test_shape.cpp
	tensor/test_strides.cpp
	tensor/test_storage_offset.cpp
	tensor/test_tensor_metadata.cpp
	tensor/test_device.cpp
	tensor/test_dtype.cpp
	tensor/test_layout.cpp
	tensor/test_tensor.cpp
	tensor/test_indexing.cpp
)

add_executable(
	tensor_view_tests
	
	metadata/test_reshape.cpp
	metadata/test_flatten.cpp
	metadata/test_transpose.cpp
	metadata/test_permute.cpp
)

target_link_libraries(

    storage_tests

    PRIVATE

    nova
    GTest::gtest
    GTest::gtest_main

)

target_link_libraries(

    allocator_tests

    PRIVATE

    nova
    GTest::gtest
    GTest::gtest_main

)

target_link_libraries(

    tensor_tests

    PRIVATE

    nova
    GTest::gtest
    GTest::gtest_main

)

target_link_libraries(

    tensor_view_tests

    PRIVATE

    nova
    GTest::gtest
    GTest::gtest_main

)

include(GoogleTest)

gtest_discover_tests(allocator_tests)
gtest_discover_tests(storage_tests)
gtest_discover_tests(tensor_tests)
gtest_discover_tests(tensor_view_tests)