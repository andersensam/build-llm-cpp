# build-llm-cpp

An exploration of building a large language model in C++.

## How to build

This project uses `bazel` to build. For debugging purposes, build with:
```
bazel build //... --config debug
```

Using the debug mode disables optimizations, turns on debug symbols, and uses address sanitizer checks.

`build-llm-cpp` will be produced in `bazel-bin/build-llm-cpp`.