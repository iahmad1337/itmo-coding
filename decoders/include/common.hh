#include <memory>
#include <vector>

using u64 = uint64_t;
using i64 = int64_t;

using TMatrix = int;
using TBinaryVector = int;  // TODO: bitmap/vector<bool> wrapper with arithmetic

template<class T>
using TVector = std::vector<T>;

struct TCode {
  TBinaryVector Encode(TBinaryVector in);


  const u64 n;
  const u64 k;
  TMatrix GeneratorMatrix;
};

struct TDecoder {
  TBinaryVector Decode(TVector<double> in);

  std::shared_ptr<TCode> Code;
};

/// 2-AM AWGN
/// https://en.wikipedia.org/wiki/Signal-to-noise_ratio#Amplitude_modulation
struct TSimulator {

  /// Return the frequency of errors
  double Simulate(double noiseLevel, u64 iterations, u64 maxErrors);

  std::shared_ptr<TCode> Code;
  std::shared_ptr<TDecoder> Decoder;
};

int main() {
}

