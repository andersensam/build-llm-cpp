# benchmarks

Below are benchmark results for various libraries in build-llm-cpp.

## Matrix Multiplication Performance Benchmarks & Analysis

### Complete Raw Benchmark Data

| Date | Matmul Impl | Compile Flag | Operands | Out Shape | Tensor Types | Dtype | Time (ms) | Weight (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: |
| 10/7/26 | Naïve v1 | DEBUG | [256, 512] @ [512, 256] | [256, 256] | Tensor, Tensor | int16_t (short) | 10430 | 18.60% |
| 10/7/26 | Naïve v1 | DEBUG | [256, 512] @ [512, 256] | [256, 256] | Tensor, Tensor | int64_t (long long) | 10190 | 18.20% |
| 10/7/26 | Naïve v1 | DEBUG | [256, 512] @ [512, 256] | [256, 256] | Tensor, Tensor | uint16_t (unsigned short) | 9860 | 17.60% |
| 10/7/26 | Naïve v1 | DEBUG | [256, 512] @ [512, 256] | [256, 256] | Tensor, Tensor | float (float32) | 9420 | 16.80% |
| 10/7/26 | Naïve v1 | DEBUG | [128, 512] @ [64, 512]^T | [128, 64] | ListTensorSlice, ListTensorSlice | float (float32) | 13460 | 24.00% |
| 10/7/26 | Naïve v1 | DEBUG | [128, 256]^T @ [128, 256] | [256, 256] | Tensor, Tensor | double (float64) | 2580 | 4.60% |
| 10/7/26 | Naïve v1 | VECTORIZE | [256, 512] @ [512, 256] | [256, 256] | Tensor, Tensor | int16_t (short) | 308 | 19.10% |
| 10/7/26 | Naïve v1 | VECTORIZE | [256, 512] @ [512, 256] | [256, 256] | Tensor, Tensor | int64_t (long long) | 296 | 18.40% |
| 10/7/26 | Naïve v1 | VECTORIZE | [256, 512] @ [512, 256] | [256, 256] | Tensor, Tensor | uint16_t (unsigned short) | 282 | 17.50% |
| 10/7/26 | Naïve v1 | VECTORIZE | [256, 512] @ [512, 256] | [256, 256] | Tensor, Tensor | float (float32) | 276 | 17.10% |
| 10/7/26 | Naïve v1 | VECTORIZE | [128, 512] @ [64, 512]^T | [128, 64] | ListTensorSlice, ListTensorSlice | float (float32) | 362 | 22.50% |
| 10/7/26 | Naïve v1 | VECTORIZE | [128, 256]^T @ [128, 256] | [256, 256] | Tensor, Tensor | double (float64) | 76 | 4.70% |
| 10/7/26 | Naïve v2 | DEBUG | [256, 512] @ [512, 256] | [256, 256] | Tensor, Tensor | int16_t (short) | 259 | 13.50% |
| 10/7/26 | Naïve v2 | DEBUG | [256, 512] @ [512, 256] | [256, 256] | Tensor, Tensor | int64_t (long long) | 304 | 15.80% |
| 10/7/26 | Naïve v2 | DEBUG | [256, 512] @ [512, 256] | [256, 256] | Tensor, Tensor | uint16_t (unsigned short) | 713 | 37.20% |
| 10/7/26 | Naïve v2 | DEBUG | [256, 512] @ [512, 256] | [256, 256] | Tensor, Tensor | float (float32) | 192 | 10.00% |
| 10/7/26 | Naïve v2 | DEBUG | [128, 512] @ [64, 512]^T | [128, 64] | ListTensorSlice, ListTensorSlice | float (float32) | 289 | 15.10% |
| 10/7/26 | Naïve v2 | DEBUG | [128, 256]^T @ [128, 256] | [256, 256] | Tensor, Tensor | double (float64) | 50 | 2.60% |
| 10/7/26 | Naïve v2 | VECTORIZE | [256, 512] @ [512, 256] | [256, 256] | Tensor, Tensor | int16_t (short) | 3 | 7.30% |
| 10/7/26 | Naïve v2 | VECTORIZE | [256, 512] @ [512, 256] | [256, 256] | Tensor, Tensor | int64_t (long long) | 15 | 36.60% |
| 10/7/26 | Naïve v2 | VECTORIZE | [256, 512] @ [512, 256] | [256, 256] | Tensor, Tensor | uint16_t (unsigned short) | 4 | 9.80% |
| 10/7/26 | Naïve v2 | VECTORIZE | [256, 512] @ [512, 256] | [256, 256] | Tensor, Tensor | float (float32) | 2 | 4.90% |
| 10/7/26 | Naïve v2 | VECTORIZE | [128, 512] @ [64, 512]^T | [128, 64] | ListTensorSlice, ListTensorSlice | float (float32) | 5 | 12.20% |
| 10/7/26 | Naïve v2 | VECTORIZE | [128, 256]^T @ [128, 256] | [256, 256] | Tensor, Tensor | double (float64) | 1 | 2.40% |

---

### Compiler Optimization Speedups (DEBUG vs. VECTORIZE)

Speedups are calculated as `DEBUG Time (ms) / VECTORIZE Time (ms)`.

#### Naïve v1

| Operands | Dtype | DEBUG (ms) | VECTORIZE (ms) | Speedup |
| :--- | :--- | ---: | ---: | ---: |
| [256, 512] @ [512, 256] | int16_t (short) | 10,430 | 308 | **33.86x** |
| [256, 512] @ [512, 256] | int64_t (long long) | 10,190 | 296 | **34.43x** |
| [256, 512] @ [512, 256] | uint16_t (unsigned short) | 9,860 | 282 | **34.96x** |
| [256, 512] @ [512, 256] | float (float32) | 9,420 | 276 | **34.13x** |
| [128, 512] @ [64, 512]^T | float (float32) | 13,460 | 362 | **37.18x** |
| [128, 256]^T @ [128, 256] | double (float64) | 2,580 | 76 | **33.95x** |

#### Naïve v2

| Operands | Dtype | DEBUG (ms) | VECTORIZE (ms) | Speedup |
| :--- | :--- | ---: | ---: | ---: |
| [256, 512] @ [512, 256] | int16_t (short) | 259 | 3 | **86.33x** |
| [256, 512] @ [512, 256] | int64_t (long long) | 304 | 15 | **20.27x** |
| [256, 512] @ [512, 256] | uint16_t (unsigned short) | 713 | 4 | **178.25x** |
| [256, 512] @ [512, 256] | float (float32) | 192 | 2 | **96.00x** |
| [128, 512] @ [64, 512]^T | float (float32) | 289 | 5 | **57.80x** |
| [128, 256]^T @ [128, 256] | double (float64) | 50 | 1 | **50.00x** |

---

### Implementation Comparison (Naïve v1 vs. Naïve v2)

Speedups are calculated as `v1 Time (ms) / v2 Time (ms)`. Values above 1.0x indicate that **Naïve v2** is faster.

| Operands | Dtype | DEBUG v1 (ms) | DEBUG v2 (ms) | DEBUG Speedup (v2 over v1) | VECTORIZE v1 (ms) | VECTORIZE v2 (ms) | VECTORIZE Speedup (v2 over v1) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| [256, 512] @ [512, 256] | int16_t (short) | 10,430 | 259 | **40.27x** | 308 | 3 | **102.67x** |
| [256, 512] @ [512, 256] | int64_t (long long) | 10,190 | 304 | **33.52x** | 296 | 15 | **19.73x** |
| [256, 512] @ [512, 256] | uint16_t (unsigned short) | 9,860 | 713 | **13.83x** | 282 | 4 | **70.50x** |
| [256, 512] @ [512, 256] | float (float32) | 9,420 | 192 | **49.06x** | 276 | 2 | **138.00x** |
| [128, 512] @ [64, 512]^T | float (float32) | 13,460 | 289 | **46.57x** | 362 | 5 | **72.40x** |
| [128, 256]^T @ [128, 256] | double (float64) | 2,580 | 50 | **51.60x** | 76 | 1 | **76.00x** |

---

### Performance Highlights

* **Implementation Superiority (v2 vs v1):**
  * **DEBUG Mode:** Naïve v2 is **39.14x faster** on average compared to Naïve v1 (ranging from **13.83x** on `uint16_t` up to **51.60x** on `double`).
  * **VECTORIZE Mode:** Naïve v2 widens the gap significantly, running **79.88x faster** on average compared to Naïve v1 (peaking at **138.00x** for standard `float32`).

* **Vectorization Efficiency:**
  * Naïve v1 shows uniform speedup across data types when vectorized (**~34x to 37x**).
  * Naïve v2 shows extreme gains under vectorization, especially for smaller element bit-widths and floating-point types (**178.25x** for `uint16_t` and **96.00x** for `float32`).

* **Data Type & Memory Access Effects:**
  * **int64_t (long long)** exhibits the lowest vectorization gains in v2 (**20.27x** internal speedup; **19.73x** improvement over v1), caused by wider 64-bit element sizes filling SIMD vector registers faster.
  * **ListTensorSlice** operands introduce minor overhead across both implementations, but v2 maintains a **72.40x** advantage over v1 when vectorized.