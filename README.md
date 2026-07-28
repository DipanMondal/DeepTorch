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
---
## Tensor Design
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
```