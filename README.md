# build-llm-cpp

An exploration of building a large language model in C++.

## How to build

This project uses `bazel` to build. For debugging purposes, build with:
```
bazel build //... --config debug
```

Using the debug mode disables optimizations, turns on debug symbols, and uses address sanitizer checks.

`build-llm-cpp` will be produced in `bazel-bin/build-llm-cpp`.

## Unit tests

Unit tests are available in the `tests/` directory and are buildable in `bazel` via:
```
bazel test //...
```

## AI usage in this project

The primary goal of this exploration into low-level model training is to properly absorb the concepts and understand
more of the pitfalls involved in designing such a solution. As such, this project is nearly AI-free; however,
I feel compelled to disclose that unit tests and functions designated with:
```
DISCLAIMER: AI tools were used
```
in the docstring, include AI usage.

For functions without this designation (or files without my usual andersensam header), it is safe to assume
that AI was not used in the neither the design nor implementation.
