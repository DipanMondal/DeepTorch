# Deep Torch
---
## Memory Architecture
```
                Tensor

                   │

             shared_ptr

                   │

              Storage

                   │

             Allocator

          ┌───────────────┐
          │               │
    CPUAllocator    CUDAAllocator
          │               │
      malloc()      cudaMalloc()
```
### Responsibilities
#### Allocator
Responsible only for allocating and freeing memory.
- allocate()
- deallocate()
Nothing else.
It also handles the memory location (CPU or GPU).

#### Storage
Responsible for owning memory.
Contains:
- pointer
- size
- allocator

---
## Tensor Design
#### Tensor
Tensor owns the Storage and other metadata about the storage. It is an abstraction of the 
underlying memory and doesn't need to concern about Raw Memory.
```
Tensor
│
├── Shape
├── Strides
├── DType
├── Device
├── Offset
└── Storage ─────────────┐
                         │
                         ▼
                  Raw Memory
				  
class Tensor
{
private:
	TensorMetadata metadata_;
	shared_ptr<Storage> storage_;
}
```

#### TensorMetadata
```
class TensorMetadata
{
private:
    Shape shape_;
    Strides strides_;
    StorageOffset offset_;
    Device device_;
    DType dtype_;
    Layout layout_;
};
```