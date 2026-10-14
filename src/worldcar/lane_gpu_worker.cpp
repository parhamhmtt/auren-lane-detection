#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {
constexpr int kWidth = 320;
constexpr int kHeight = 180;
constexpr int kPixels = kWidth * kHeight;

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR) std::cerr << message << '\n';
  }
};

bool read_exact(std::istream& input, char* data, std::streamsize size) {
  input.read(data, size);
  return input.gcount() == size;
}

std::size_t volume(const nvinfer1::Dims& dims) {
  std::size_t result = 1;
  for (int i = 0; i < dims.nbDims; ++i) result *= static_cast<std::size_t>(dims.d[i]);
  return result;
}
}
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream file(argv[1], std::ios::binary);
  if (!file) return 3;
  std::vector<char> engine_data((std::istreambuf_iterator<char>(file)), {});
  Logger logger;
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
  if (!runtime) return 4;
  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(
      engine_data.data(), engine_data.size(), nullptr);
  if (!engine) return 5;
  nvinfer1::IExecutionContext* context = engine->createExecutionContext();
  if (!context || engine->getNbBindings() != 2) return 6;

  int input_index = -1;
  int output_index = -1;
  for (int i = 0; i < engine->getNbBindings(); ++i) {
    (engine->bindingIsInput(i) ? input_index : output_index) = i;
  }
  if (input_index < 0 || output_index < 0) return 7;
  const std::size_t input_count = volume(engine->getBindingDimensions(input_index));
  const std::size_t output_count = volume(engine->getBindingDimensions(output_index));
  if (input_count != static_cast<std::size_t>(3 * kPixels) ||
      output_count != static_cast<std::size_t>(3 * kPixels)) return 8;

  std::vector<std::uint8_t> bgr(3 * kPixels);
  std::vector<float> input(input_count);
  std::vector<float> output(output_count);
  std::vector<std::uint8_t> labels(kPixels);
  void* bindings[2] = {nullptr, nullptr};
  if (cudaMalloc(&bindings[input_index], input_count * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&bindings[output_index], output_count * sizeof(float)) != cudaSuccess) return 9;
  cudaStream_t stream;
  if (cudaStreamCreate(&stream) != cudaSuccess) return 10;

  std::cout.put('G');
  std::cout.flush();
  while (read_exact(std::cin, reinterpret_cast<char*>(bgr.data()), bgr.size())) {
    for (int pixel = 0; pixel < kPixels; ++pixel) {
      input[pixel] = static_cast<float>(bgr[3 * pixel + 2]) / 255.0F;
      input[kPixels + pixel] = static_cast<float>(bgr[3 * pixel + 1]) / 255.0F;
      input[2 * kPixels + pixel] = static_cast<float>(bgr[3 * pixel]) / 255.0F;
    }
    if (cudaMemcpyAsync(bindings[input_index], input.data(),
                        input_count * sizeof(float), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess ||
        !context->enqueueV2(bindings, stream, nullptr) ||
        cudaMemcpyAsync(output.data(), bindings[output_index],
                        output_count * sizeof(float), cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) return 11;
    for (int pixel = 0; pixel < kPixels; ++pixel) {
      std::uint8_t best = 0;
      if (output[kPixels + pixel] > output[pixel]) best = 1;
      if (output[2 * kPixels + pixel] > output[best * kPixels + pixel]) best = 2;
      labels[pixel] = best;
    }
    std::cout.write(reinterpret_cast<const char*>(labels.data()), labels.size());
    std::cout.flush();
  }

  cudaStreamDestroy(stream);
  cudaFree(bindings[input_index]);
  cudaFree(bindings[output_index]);
  context->destroy();
  engine->destroy();
  runtime->destroy();
  return 0;
}
