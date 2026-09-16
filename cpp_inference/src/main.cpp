#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wincodec.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>

namespace {

constexpr int kResizeShortSide = 256;
constexpr int kCropSize = 224;
constexpr std::array<float, 3> kMean = {0.485F, 0.456F, 0.406F};
constexpr std::array<float, 3> kStd = {0.229F, 0.224F, 0.225F};

struct Options {
  std::filesystem::path model;
  std::filesystem::path image;
  std::filesystem::path input_bin;
  std::filesystem::path labels;
  std::filesystem::path dump_input;
  std::filesystem::path dump_logits;
  std::filesystem::path benchmark_output;
  int warmup_iterations = -1;
  int measured_iterations = -1;
  int repeat_id = -1;
  int run_order = -1;
  int expected_top1 = 285;

  bool IsBenchmark() const { return !benchmark_output.empty(); }
};

struct RgbImage {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<uint8_t> pixels;
};

template <typename T>
class ComPtr {
 public:
  ComPtr() = default;
  ~ComPtr() { Reset(); }
  ComPtr(const ComPtr&) = delete;
  ComPtr& operator=(const ComPtr&) = delete;

  T* Get() const { return ptr_; }
  T** Put() {
    Reset();
    return &ptr_;
  }

 private:
  void Reset() {
    if (ptr_ != nullptr) {
      ptr_->Release();
      ptr_ = nullptr;
    }
  }

  T* ptr_ = nullptr;
};

class ComApartment {
 public:
  ComApartment() {
    const HRESULT result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(result)) {
      throw std::runtime_error("CoInitializeEx failed with HRESULT " + HresultToString(result));
    }
    initialized_ = true;
  }
  ~ComApartment() {
    if (initialized_) {
      CoUninitialize();
    }
  }

 private:
  static std::string HresultToString(HRESULT result) {
    std::ostringstream stream;
    stream << "0x" << std::hex << std::uppercase << static_cast<unsigned long>(result);
    return stream.str();
  }

  bool initialized_ = false;
};

std::string HresultToString(HRESULT result) {
  std::ostringstream stream;
  stream << "0x" << std::hex << std::uppercase << static_cast<unsigned long>(result);
  return stream.str();
}

void CheckHr(HRESULT result, const std::string& operation) {
  if (FAILED(result)) {
    throw std::runtime_error(operation + " failed with HRESULT " + HresultToString(result));
  }
}

void PrintUsage(const char* program) {
  std::cout << "Usage: " << program
            << " --model <model.onnx> (--image <image.jpg> | --input-bin <tensor.f32>)"
            << " --labels <imagenet_classes.txt> [--dump-input <tensor.f32>]"
            << " [--dump-logits <logits.f32>] [--benchmark-output <worker.csv>"
            << " --warmup 50 --runs 300 --repeat-id <id> --run-order <id>]\n";
}

int ParseNonnegativeInt(const char* value, const std::string& flag) {
  try {
    const int parsed = std::stoi(value);
    if (parsed < 0) {
      throw std::runtime_error("negative value");
    }
    return parsed;
  } catch (const std::exception&) {
    throw std::runtime_error("Expected a non-negative integer for " + flag);
  }
}

Options ParseOptions(int argc, char* argv[]) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    const std::string flag = argv[index];
    if (flag == "--help" || flag == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::runtime_error("Missing value for " + flag);
    }
    if (flag == "--model") {
      options.model = argv[index + 1];
    } else if (flag == "--image") {
      options.image = argv[index + 1];
    } else if (flag == "--input-bin") {
      options.input_bin = argv[index + 1];
    } else if (flag == "--labels") {
      options.labels = argv[index + 1];
    } else if (flag == "--dump-input") {
      options.dump_input = argv[index + 1];
    } else if (flag == "--dump-logits") {
      options.dump_logits = argv[index + 1];
    } else if (flag == "--benchmark-output") {
      options.benchmark_output = argv[index + 1];
    } else if (flag == "--warmup") {
      options.warmup_iterations = ParseNonnegativeInt(argv[index + 1], flag);
    } else if (flag == "--runs") {
      options.measured_iterations = ParseNonnegativeInt(argv[index + 1], flag);
    } else if (flag == "--repeat-id") {
      options.repeat_id = ParseNonnegativeInt(argv[index + 1], flag);
    } else if (flag == "--run-order") {
      options.run_order = ParseNonnegativeInt(argv[index + 1], flag);
    } else if (flag == "--expected-top1") {
      options.expected_top1 = ParseNonnegativeInt(argv[index + 1], flag);
    } else {
      throw std::runtime_error("Unknown argument: " + flag);
    }
  }

  if (options.model.empty() || options.labels.empty()) {
    throw std::runtime_error("--model and --labels are required");
  }
  if (options.image.empty() == options.input_bin.empty()) {
    throw std::runtime_error("Provide exactly one of --image or --input-bin");
  }
  if (!options.dump_input.empty() && options.image.empty()) {
    throw std::runtime_error("--dump-input is only valid with --image");
  }
  if (options.IsBenchmark()) {
    if (options.input_bin.empty() || !options.image.empty()) {
      throw std::runtime_error("Benchmark mode requires --input-bin and forbids --image");
    }
    if (options.warmup_iterations != 50 || options.measured_iterations != 300 || options.repeat_id < 0 ||
        options.run_order < 0) {
      throw std::runtime_error("Frozen S4 benchmark requires --warmup 50 --runs 300 --repeat-id and --run-order");
    }
  }
  return options;
}

RgbImage LoadJpegRgb(const std::filesystem::path& image_path) {
  if (!std::filesystem::exists(image_path)) {
    throw std::runtime_error("Image does not exist: " + image_path.string());
  }

  ComApartment apartment;
  ComPtr<IWICImagingFactory> factory;
  CheckHr(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                           __uuidof(IWICImagingFactory), reinterpret_cast<void**>(factory.Put())),
          "Create WIC imaging factory");

  ComPtr<IWICBitmapDecoder> decoder;
  CheckHr(factory.Get()->CreateDecoderFromFilename(image_path.wstring().c_str(), nullptr, GENERIC_READ,
                                                    WICDecodeMetadataCacheOnDemand, decoder.Put()),
          "Decode JPEG file");

  ComPtr<IWICBitmapFrameDecode> frame;
  CheckHr(decoder.Get()->GetFrame(0, frame.Put()), "Read JPEG frame");

  UINT width = 0;
  UINT height = 0;
  CheckHr(frame.Get()->GetSize(&width, &height), "Read JPEG dimensions");
  if (width == 0 || height == 0) {
    throw std::runtime_error("JPEG has an invalid zero dimension");
  }

  ComPtr<IWICFormatConverter> converter;
  CheckHr(factory.Get()->CreateFormatConverter(converter.Put()), "Create WIC RGB converter");
  CheckHr(converter.Get()->Initialize(frame.Get(), GUID_WICPixelFormat24bppRGB,
                                       WICBitmapDitherTypeNone, nullptr, 0.0,
                                       WICBitmapPaletteTypeCustom),
          "Convert JPEG to 24-bit RGB");

  const UINT stride = width * 3;
  RgbImage image{width, height, std::vector<uint8_t>(static_cast<size_t>(stride) * height)};
  CheckHr(converter.Get()->CopyPixels(nullptr, stride,
                                      static_cast<UINT>(image.pixels.size()), image.pixels.data()),
          "Copy JPEG RGB pixels");
  return image;
}

RgbImage ResizeBilinear(const RgbImage& source, uint32_t target_width, uint32_t target_height) {
  RgbImage target{target_width, target_height,
                  std::vector<uint8_t>(static_cast<size_t>(target_width) * target_height * 3)};
  const float scale_x = static_cast<float>(source.width) / target_width;
  const float scale_y = static_cast<float>(source.height) / target_height;

  for (uint32_t y = 0; y < target_height; ++y) {
    const float source_y = (static_cast<float>(y) + 0.5F) * scale_y - 0.5F;
    const int y0 = std::clamp(static_cast<int>(std::floor(source_y)), 0, static_cast<int>(source.height) - 1);
    const int y1 = std::min(y0 + 1, static_cast<int>(source.height) - 1);
    const float wy = source_y - std::floor(source_y);
    for (uint32_t x = 0; x < target_width; ++x) {
      const float source_x = (static_cast<float>(x) + 0.5F) * scale_x - 0.5F;
      const int x0 = std::clamp(static_cast<int>(std::floor(source_x)), 0, static_cast<int>(source.width) - 1);
      const int x1 = std::min(x0 + 1, static_cast<int>(source.width) - 1);
      const float wx = source_x - std::floor(source_x);
      for (int channel = 0; channel < 3; ++channel) {
        const auto pixel = [&source, channel](int px, int py) {
          return static_cast<float>(source.pixels[(static_cast<size_t>(py) * source.width + px) * 3 + channel]);
        };
        const float top = pixel(x0, y0) * (1.0F - wx) + pixel(x1, y0) * wx;
        const float bottom = pixel(x0, y1) * (1.0F - wx) + pixel(x1, y1) * wx;
        target.pixels[(static_cast<size_t>(y) * target_width + x) * 3 + channel] =
            static_cast<uint8_t>(std::clamp(std::round(top * (1.0F - wy) + bottom * wy), 0.0F, 255.0F));
      }
    }
  }
  return target;
}

std::vector<float> PrepareResNet18Input(const RgbImage& decoded) {
  const float resize_scale = static_cast<float>(kResizeShortSide) /
                             static_cast<float>(std::min(decoded.width, decoded.height));
  const uint32_t resized_width = static_cast<uint32_t>(std::round(decoded.width * resize_scale));
  const uint32_t resized_height = static_cast<uint32_t>(std::round(decoded.height * resize_scale));
  const RgbImage resized = ResizeBilinear(decoded, resized_width, resized_height);

  if (resized.width < kCropSize || resized.height < kCropSize) {
    throw std::runtime_error("Resized image is smaller than center crop");
  }
  const uint32_t left = (resized.width - kCropSize) / 2;
  const uint32_t top = (resized.height - kCropSize) / 2;

  std::vector<float> tensor(3 * kCropSize * kCropSize);
  for (int channel = 0; channel < 3; ++channel) {
    for (int y = 0; y < kCropSize; ++y) {
      for (int x = 0; x < kCropSize; ++x) {
        const size_t source_offset =
            (static_cast<size_t>(top + y) * resized.width + (left + x)) * 3 + channel;
        const size_t tensor_offset = static_cast<size_t>(channel) * kCropSize * kCropSize + y * kCropSize + x;
        const float unit_rgb = static_cast<float>(resized.pixels[source_offset]) / 255.0F;
        tensor[tensor_offset] = (unit_rgb - kMean[channel]) / kStd[channel];
      }
    }
  }

  std::cout << "Preprocess: WIC JPEG decode -> RGB -> resize shorter side to " << kResizeShortSide
            << " (bilinear, aspect preserved) -> center crop " << kCropSize << "x" << kCropSize
            << " -> [0,1] -> ImageNet normalize -> NCHW\n";
  std::cout << "Decoded RGB size: " << decoded.width << "x" << decoded.height << "\n";
  return tensor;
}

void PrintInputStats(const std::vector<float>& tensor, const std::string& source) {
  constexpr size_t kInputElementCount = 3 * kCropSize * kCropSize;
  if (tensor.size() != kInputElementCount) {
    throw std::runtime_error("Input tensor has an invalid element count");
  }
  std::cout << "Input source: " << source << "\n";
  std::cout << "Input tensor shape: [1, 3, " << kCropSize << ", " << kCropSize << "]\n";
  const auto [min_it, max_it] = std::minmax_element(tensor.begin(), tensor.end());
  const double mean = std::accumulate(tensor.begin(), tensor.end(), 0.0) / tensor.size();
  std::cout << std::fixed << std::setprecision(6) << "Normalized input stats: min=" << *min_it
            << ", max=" << *max_it << ", mean=" << mean << "\n";
}

std::vector<float> ReadFloat32Tensor(const std::filesystem::path& input_path, size_t expected_count) {
  if (!std::filesystem::exists(input_path)) {
    throw std::runtime_error("Input tensor file does not exist: " + input_path.string());
  }
  const uintmax_t expected_bytes = static_cast<uintmax_t>(expected_count) * sizeof(float);
  const uintmax_t actual_bytes = std::filesystem::file_size(input_path);
  if (actual_bytes != expected_bytes) {
    throw std::runtime_error("Input tensor byte size mismatch: expected " + std::to_string(expected_bytes) +
                             ", got " + std::to_string(actual_bytes));
  }
  std::ifstream input(input_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Cannot open input tensor file: " + input_path.string());
  }
  std::vector<float> values(expected_count);
  input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(expected_bytes));
  if (input.gcount() != static_cast<std::streamsize>(expected_bytes) || input.bad()) {
    throw std::runtime_error("Failed to read the complete float32 tensor");
  }
  return values;
}

void WriteFloat32Tensor(const std::filesystem::path& output_path, const float* values, size_t count,
                        const std::string& description) {
  if (output_path.empty()) {
    return;
  }
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("Cannot open " + description + " output: " + output_path.string());
  }
  const std::streamsize byte_count = static_cast<std::streamsize>(count * sizeof(float));
  output.write(reinterpret_cast<const char*>(values), byte_count);
  if (!output) {
    throw std::runtime_error("Failed to write " + description + " output: " + output_path.string());
  }
  std::cout << "Wrote " << description << ": " << output_path.string() << " (" << count << " float32 values)\n";
}

void SetAndVerifyAffinityCpu0() {
  constexpr DWORD_PTR kCpu0Mask = 1;
  if (!SetProcessAffinityMask(GetCurrentProcess(), kCpu0Mask)) {
    throw std::runtime_error("SetProcessAffinityMask(logical CPU 0) failed with error " +
                             std::to_string(GetLastError()));
  }
  DWORD_PTR process_mask = 0;
  DWORD_PTR system_mask = 0;
  if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
    throw std::runtime_error("GetProcessAffinityMask verification failed with error " +
                             std::to_string(GetLastError()));
  }
  if (process_mask != kCpu0Mask) {
    throw std::runtime_error("Affinity verification failed: expected process mask 1, got " +
                             std::to_string(static_cast<unsigned long long>(process_mask)));
  }
  std::cout << "Affinity verified: logical CPU 0 (process mask=1)\n";
}

size_t Argmax(const float* values, size_t count) {
  return static_cast<size_t>(std::distance(values, std::max_element(values, values + count)));
}

std::vector<std::string> LoadLabels(const std::filesystem::path& labels_path) {
  std::ifstream input(labels_path);
  if (!input) {
    throw std::runtime_error("Cannot open labels file: " + labels_path.string());
  }
  std::vector<std::string> labels;
  for (std::string line; std::getline(input, line);) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    labels.push_back(line);
  }
  if (labels.size() != 1000) {
    throw std::runtime_error("Expected exactly 1000 ImageNet labels, got " + std::to_string(labels.size()));
  }
  return labels;
}

std::string ShapeToString(const std::vector<int64_t>& shape) {
  std::ostringstream stream;
  stream << "[";
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) stream << ", ";
    stream << shape[index];
  }
  stream << "]";
  return stream.str();
}

void PrintModelIo(const Ort::Session& session) {
  Ort::AllocatorWithDefaultOptions allocator;
  const auto print_side = [&session, &allocator](const char* side, size_t count, bool is_input) {
    std::cout << side << " count: " << count << "\n";
    for (size_t index = 0; index < count; ++index) {
      auto name = is_input ? session.GetInputNameAllocated(index, allocator)
                           : session.GetOutputNameAllocated(index, allocator);
      const auto type_info = is_input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
      const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      std::cout << "  " << side << "[" << index << "] name=" << name.get()
                << ", element_type=" << static_cast<int>(tensor_info.GetElementType())
                << ", shape=" << ShapeToString(tensor_info.GetShape()) << "\n";
    }
  };
  print_side("Input", session.GetInputCount(), true);
  print_side("Output", session.GetOutputCount(), false);
}

void PrintTop5(const float* logits, size_t count, const std::vector<std::string>& labels) {
  if (count != labels.size()) {
    throw std::runtime_error("Output size and labels size do not match");
  }
  std::vector<size_t> indices(count);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(),
                    [logits](size_t left, size_t right) { return logits[left] > logits[right]; });
  const float max_logit = *std::max_element(logits, logits + count);
  double denominator = 0.0;
  for (size_t index = 0; index < count; ++index) {
    denominator += std::exp(static_cast<double>(logits[index] - max_logit));
  }

  std::cout << "Top-5 predictions:\n";
  std::cout << std::fixed << std::setprecision(4);
  for (size_t rank = 0; rank < 5; ++rank) {
    const size_t index = indices[rank];
    const double probability = std::exp(static_cast<double>(logits[index] - max_logit)) / denominator;
    std::cout << "  " << (rank + 1) << ". index=" << index << ", label=" << labels[index]
              << ", probability=" << probability << "\n";
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const Options options = ParseOptions(argc, argv);
    if (options.IsBenchmark()) {
      SetAndVerifyAffinityCpu0();
      std::cout << "S4 worker configuration: threads=1, sequential execution, ORT_ENABLE_ALL, batch=1\n";
    }
    const std::vector<std::string> labels = LoadLabels(options.labels);
    constexpr size_t kInputElementCount = 3 * kCropSize * kCropSize;
    std::vector<float> input;
    if (!options.image.empty()) {
      const RgbImage decoded = LoadJpegRgb(options.image);
      input = PrepareResNet18Input(decoded);
      WriteFloat32Tensor(options.dump_input, input.data(), input.size(), "preprocessed input tensor");
      PrintInputStats(input, "C++ WIC JPEG preprocessing");
    } else {
      input = ReadFloat32Tensor(options.input_bin, kInputElementCount);
      PrintInputStats(input, "canonical --input-bin float32 tensor");
    }

    std::cout << "ONNX Runtime version: " << OrtGetApiBase()->GetVersionString() << "\n";
    const auto providers = Ort::GetAvailableProviders();
    std::cout << "Available providers:";
    for (const auto& provider : providers) std::cout << " " << provider;
    std::cout << "\n";

    Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "resnet18_ort_cpp");
    Ort::SessionOptions session_options;
    session_options.AppendExecutionProvider_CPU(1);
    if (options.IsBenchmark()) {
      session_options.SetIntraOpNumThreads(1);
      session_options.SetInterOpNumThreads(1);
      session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
      session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }
    std::cout << "Session provider: CPUExecutionProvider (explicitly appended)\n";
    Ort::Session session(environment, options.model.c_str(), session_options);
    PrintModelIo(session);

    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
      throw std::runtime_error("This S2 skeleton expects one model input and one model output");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const std::array<int64_t, 4> input_shape = {1, 3, kCropSize, kCropSize};
    const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input.data()),
                                                                input.size(), input_shape.data(), input_shape.size());
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};
    const auto run_once = [&session, &input_names, &input_tensor, &output_names]() {
      return session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    };
    const std::vector<Ort::Value> outputs = run_once();

    const auto output_info = outputs.at(0).GetTensorTypeAndShapeInfo();
    const auto output_shape = output_info.GetShape();
    const size_t output_count = output_info.GetElementCount();
    if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || output_count != labels.size()) {
      throw std::runtime_error("Expected exactly 1000 float32 output logits");
    }
    std::cout << "Runtime output tensor shape: " << ShapeToString(output_shape)
              << ", element_count=" << output_count << "\n";
    const float* logits = outputs.at(0).GetTensorData<float>();
    WriteFloat32Tensor(options.dump_logits, logits, output_count, "output logits");
    PrintTop5(logits, output_count, labels);
    if (Argmax(logits, output_count) != static_cast<size_t>(options.expected_top1)) {
      throw std::runtime_error("Correctness check failed: unexpected Top-1 index " +
                               std::to_string(Argmax(logits, output_count)));
    }

    if (options.IsBenchmark()) {
      for (int iteration = 0; iteration < options.warmup_iterations; ++iteration) {
        const auto warmup_output = run_once();
        if (warmup_output.at(0).GetTensorTypeAndShapeInfo().GetElementCount() != output_count) {
          throw std::runtime_error("Warm-up output element count changed");
        }
      }
      std::ofstream benchmark_output(options.benchmark_output, std::ios::trunc);
      if (!benchmark_output) {
        throw std::runtime_error("Cannot open benchmark output: " + options.benchmark_output.string());
      }
      benchmark_output << "repeat_id,iteration,latency_ms\n";
      benchmark_output << std::fixed << std::setprecision(9);
      for (int iteration = 0; iteration < options.measured_iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        const auto measured_output = run_once();
        const auto end = std::chrono::steady_clock::now();
        if (measured_output.at(0).GetTensorTypeAndShapeInfo().GetElementCount() != output_count) {
          throw std::runtime_error("Measured output element count changed");
        }
        const double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
        benchmark_output << options.repeat_id << ',' << iteration << ',' << latency_ms << '\n';
      }
      if (!benchmark_output) {
        throw std::runtime_error("Failed to write complete C++ benchmark CSV");
      }
      std::cout << "Wrote S4 C++ worker CSV: " << options.benchmark_output.string() << " ("
                << options.measured_iterations << " measured iterations, run_order=" << options.run_order << ")\n";
    }
    return 0;
  } catch (const Ort::Exception& error) {
    std::cerr << "ONNX Runtime error: " << error.what() << "\n";
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
  }
  return 1;
}
