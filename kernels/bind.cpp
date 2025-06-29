#include <torch/extension.h>

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor mask);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Flash attention forward",
          py::arg("Q"), py::arg("K"), py::arg("V"), 
          py::arg("mask"));
}