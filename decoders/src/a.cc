#include <vector>
#include <cassert>
#include <cstdint>
#include <optional>
#include <limits>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>

#if defined(__x86_64__)
#include <x86intrin.h>
#endif

// TODO: using vector = small_vector<u64, 64>; // 64 bytes inplace-storage

#ifdef TEST
// automatic backtrace printing
#include "backward.hpp"
backward::SignalHandling sh;

#define debug(action) { action; }
#else
#define debug(action) {}
#endif

using std::endl;
using std::cin;
using std::cout;

using i32 = int32_t;
using i64 = int64_t;
using u32 = uint32_t;
using u64 = uint64_t;

#define all(x) (x).begin(), (x).end()

#define SUB(i) ((i) & 0x3f)
#define UPP(i) ((i) / 64)
#define BIT(i) (1ull << (i))

constexpr static double BIT_TO_SIGNAL[2] = {1, -1};

void TODO(std::string message) {
    throw std::runtime_error("Not Implemented: " + message);
}

/// Get position of first set bit (a.k.a. NLZ)
std::optional<int> bsf(u64 limb) {
    if (limb == 0) return std::nullopt;
#if defined(__x86_64__)
    return __bsfq(limb);
#else
    int i = 0;
    while ((limb & BIT(i)) == 0) {
        i++;
    }
    return i;
#endif
}

/// Get most significant set bit
std::optional<int> bsr(u64 limb) {
    if (limb == 0) return std::nullopt;
#if defined(__x86_64__)
    return __bsrq(limb);
#else
    int i = 63;
    while ((limb & BIT(i)) == 0) {
        i--;
    }
    return i;
#endif
}

u64 popcnt(u64 limb) {
    static const unsigned char BitsSetTable256[256] = {
#define B2(n) n,     n+1,     n+1,     n+2
#define B4(n) B2(n), B2(n+1), B2(n+1), B2(n+2)
#define B6(n) B4(n), B4(n+1), B4(n+1), B4(n+2)
        B6(0), B6(1), B6(1), B6(2)
    };
#undef B2
#undef B4
#undef B6
    uint8_t* p = reinterpret_cast<uint8_t*>(&limb);
    return
        BitsSetTable256[p[0]]
        + BitsSetTable256[p[1]]
        + BitsSetTable256[p[2]]
        + BitsSetTable256[p[3]]
        + BitsSetTable256[p[4]]
        + BitsSetTable256[p[5]]
        + BitsSetTable256[p[6]]
        + BitsSetTable256[p[7]];
}

bool bit_parity(u64 limb) {
    // https://graphics.stanford.edu/~seander/bithacks.html#ParityParallel
    limb = (limb >> 32) ^ limb;
    limb = (limb >> 16) ^ limb;
    limb = (limb >> 8) ^ limb;
    limb = (limb >> 4) ^ limb;
    limb &= 0xf;

    // 110100110010110
    // compacted parity table for [0;16)
    return (0x6996 >> limb) & 1;
}

/// non-owning sequence of bits
struct BitSpan {
    void set(u64 idx, bool value) {
        assert(idx < bits);
        if (value) data[UPP(idx)] |= BIT(SUB(idx));
        else data[UPP(idx)] &= ~BIT(SUB(idx));
    }

    [[nodiscard]] bool get(u64 idx) const {
        assert(idx < bits);
        return (data[UPP(idx)] & BIT(SUB(idx))) != 0;
    }

    /// Position of first set bit
    [[nodiscard]] std::optional<u64> first_one() const {
        for (auto it = cbegin(); it != cend(); it++) {
            if (*it != 0) {
                return (it - cbegin()) * 64 + *bsf(*it);
            }
        }
        return std::nullopt;
    }

    /// Position of last set bit
    [[nodiscard]] std::optional<u64> last_one() const {
        auto it = cend();
        do {
            it--;
            if (*it != 0) {
                return ((it - cbegin()) * 64 + *bsr(*it));
            }
        } while (it != cbegin());
        return std::nullopt;
    }

    [[nodiscard]] std::string to_string() const {
        std::string result;
        result.resize(bits);
        for (u64 i = 0; i < bits; i++) {
            result[i] = get(i) ? '1' : '0';
        }
        return result;
    }

    [[nodiscard]] u64* begin() const {
        return data;
    }

    [[nodiscard]] u64* end() const {
        return data + UPP(bits + 63);
    }

    [[nodiscard]] const u64* cbegin() const {
        return const_cast<BitSpan*>(this)->begin();
    }

    [[nodiscard]] const u64* cend() const {
        return const_cast<BitSpan*>(this)->end();
    }

    BitSpan& operator^=(const BitSpan& other) {
        assert(bits == other.bits);
        const auto limbs = end() - begin();
        for (u64 i = 0; i < limbs; i++) {
            data[i] ^= other.data[i];
        }
        return *this;
    }

    [[nodiscard]] static bool ScalarProduct(const BitSpan& lhs, const BitSpan& rhs) {
        assert(lhs.bits == rhs.bits);
        u64 result = 0;
        auto lit = lhs.cbegin();
        auto rit = rhs.cbegin();
        while (lit != lhs.cend()) {
            result ^= *lit & *rit;
            lit++;
            rit++;
        }
        return bit_parity(result);
    }

    u64 *data;
    u64 bits;
};

/// owning sequence of bits
struct BitVector {
    BitVector(u64 bits) : storage(UPP(bits + 63)), bits{bits} {
        storage.assign(storage.size(), 0);
    }

    [[nodiscard]] operator BitSpan() {
        return BitSpan{storage.data(), bits};
    }

    std::vector<u64> storage;
    u64 bits;
};

bool operator==(const BitVector& lhs, const BitVector& rhs) {
    return lhs.bits == rhs.bits && lhs.storage == rhs.storage;
}

struct BitMatrix {
    BitMatrix(u64 rows, u64 columns) : rows{rows}, columns{columns}, stride{UPP(columns + 63)} {
        storage.resize(stride * rows, 0);
    }

    [[nodiscard]] BitSpan getRow(u64 i) {
        return BitSpan{&storage[stride * i], columns};
    }

    [[nodiscard]] bool get(u64 row, u64 column) const {
        return const_cast<BitMatrix*>(this)->getRow(row).get(column);
    }

    void set(u64 row, u64 column, bool value) {
        getRow(row).set(column, value);
    }

    void swap(u64 row1, u64 row2) {
        if (row1 == row2) return;
        auto span1 = getRow(row1);
        auto span2 = getRow(row2);
        auto *it1 = span1.begin();
        auto *it2 = span2.begin();
        while (it1 != span1.end()) {
            std::swap(*it1, *it2);
            it1++;
            it2++;
        }
    }

    [[nodiscard]] BitMatrix GetMinimalSpanForm() const {
        BitMatrix result = *this;
        result.DoMinimalSpanForm();
        return result;
    }

    void DoMinimalSpanForm() {
        // Do the regular gauss
        for (u64 col = 0, row = 0; col < columns && row < rows; col++) {
            // Invariant: forall i < row: col <= begin(i)
            auto row_with_one = row;
            while (row_with_one < rows && !get(row_with_one, col)) {
                row_with_one++;
            }
            if (row_with_one == rows) {
                continue;
            }

            assert(row <= row_with_one && row_with_one < rows);
            this->swap(row, row_with_one);

            for (u64 lower_row = row + 1; lower_row < rows; lower_row++) {
                if (get(lower_row, col)) {
                    getRow(lower_row) ^= getRow(row);
                    assert(!get(lower_row, col));
                }
            }
            row++;
        }  // Post: forall i: begin(i) < begin(i + 1)

        // Now, make ends different
        for (u64 row = rows - 1;; row--) {
            auto last_one = getRow(row).last_one();
            assert (bool(last_one));
            for (u64 upper_row = 0; upper_row < row; upper_row++) {
                if (get(upper_row, *last_one)) getRow(upper_row) ^= getRow(row);
            }  // Post: forall i != row: G[i][end(row)] == 0
            if (row == 0)  break;
        }  // Post: forall i != j: end(i) != end(j)
    }

    const u64 rows;
    const u64 columns;
    const u64 stride;  // number of limbs per row
    std::vector<u64> storage;
};

/*******************************************************************************
*                                  Debugging                                  *
*******************************************************************************/

template<class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "{";
    bool first = true;
    for (auto& x : v) {
        if (first) {
            os << x;
        } else {
            os << ", " << x;
        }
        first = false;
    }
    os << "}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const BitSpan& span) {
    os << "BitSpan(bits=" << span.bits << "){";
    for (u64 i = 0; i < span.bits; i++) {
        os << span.get(i);
    }
    os << "}";
    return os;
}

/*******************************************************************************
*                                   Trellis                                   *
*******************************************************************************/

constexpr inline u64 NIL = std::numeric_limits<u64>::max();
constexpr inline double INFTY = std::numeric_limits<double>::infinity();

struct Trellis {
    /// Node is mutable and so should be restored on every iteration
    struct Node {
        u64 to[2];
        bool parent_label_dp;  /// [backtracking] label of edge, from which we arived
        u64 parent_idx;        /// [backtracking] index of parent in prev layer
        double metric_dp;      /// [backtracking] best metric that we have encountered

        [[nodiscard]] static Node empty() {
            return Node{
                {NIL, NIL},
                false,
                NIL,
                INFTY,
            };
        }
    };

    struct Layer {
        std::vector<Node> nodes;
        std::vector<u64> activeRows;
        std::unordered_map<u64, u64> activeRowToIdx;

        [[nodiscard]] static Layer FromActiveRows(std::vector<u64>&& activeRows) {
            std::unordered_map<u64, u64> activeRowToIdx;
            for (u64 i = 0; i < activeRows.size(); i++) {
                activeRowToIdx[activeRows[i]] = i;
            }
            return Layer{
                std::vector<Node>((1ull << activeRows.size()), Node::empty()),
                std::move(activeRows),
                std::move(activeRowToIdx)
            };
        }
    };

    static std::vector<Layer> GetUninitializedLayersFromMSF(const BitMatrix& m) {
        const u64 N = m.columns;
        const u64 K = m.rows;

        std::vector<Layer> layers;

        layers.reserve(N + 1);

        // Dummy layer on start
        layers.emplace_back(Layer::FromActiveRows({}));

        // Calculate begin(row) & end(row)
        struct ActiveRange {
            u64 b;  // location of first 1 in row
            u64 e;  // location of last 1 in row
        };
        std::vector<ActiveRange> activeRange;
        activeRange.reserve(m.rows);
        for (u64 row_idx = 0; row_idx < m.rows; row_idx++) {
            auto rowSpan = const_cast<BitMatrix&>(m).getRow(row_idx);
            activeRange.push_back(
                ActiveRange{
                    /* b = */ *rowSpan.first_one(),
                    /* e = */ *rowSpan.last_one(),
                }
            );
        }

        for (u64 layer_idx = 0; layer_idx < N; layer_idx++) {
            std::vector<u64> activeRows;
            for (u64 row = 0; row < activeRange.size(); row++) {
                if (activeRange[row].b <= layer_idx && layer_idx < activeRange[row].e) {
                    activeRows.push_back(row);
                }
            }
            assert(std::is_sorted(activeRows.begin(), activeRows.end()));
            layers.emplace_back(
                Layer::FromActiveRows(std::move(activeRows))
            );
        }

        return layers;
    }

    /// Build a minimal trellis
    [[nodiscard]] static Trellis FromMSF(const BitMatrix& m) {
        const u64 N = m.columns;
        const u64 K = m.rows;

        Trellis t{m, {}};  // result

        // 1. Build trellis, but without any edges yet - "disjoint trellis"
        t.layers = GetUninitializedLayersFromMSF(m);

        // 2. Connect nodes with edges
        std::vector<u64> union_;  // union of active rows between layers
        for (u64 col = 0; col < N; col++) {
            // Invariant: trellis is built up until layers[col]
            union_.resize(m.rows);
            auto it = std::set_union(
                all(t.layers[col].activeRows),
                all(t.layers[col + 1].activeRows),
                union_.begin()
            );
            union_.resize(it - union_.begin());

            // scalar product calculated for edge label
            BitVector w{union_.size()};
            BitVector x{union_.size()};

            // Fill up `w` with matrix values
            for (u64 active_row_idx = 0; active_row_idx < union_.size(); active_row_idx++) {
                BitSpan(x).set(active_row_idx, m.get(union_[active_row_idx], col));
            }

            for (u64 node_idx = 0; node_idx < t.layers[col].nodes.size(); node_idx++) {
                u64 destination_idx = 0;
                for (u64 active_row_idx = 0; active_row_idx < t.layers[col].activeRows.size(); active_row_idx++) {
                    bool isActiveRowSet = (node_idx & BIT(active_row_idx)) != 0;
                    // store union of active rows into `w`
                    BitSpan(w).set(active_row_idx, isActiveRowSet);
                    // build common part of destination_idx
                    auto& prevLayer = t.layers[col];
                    auto& nextLayer = t.layers[col + 1];
                    if (
                        auto it = nextLayer.activeRowToIdx.find(prevLayer.activeRows[active_row_idx]);
                        it != nextLayer.activeRowToIdx.end()
                    ) {
                        // it->second == position of the active row in the next
                        // layer
                        destination_idx |= BIT(it->second) * isActiveRowSet;
                    }
                }

                // debug(
                //     std::cout
                //     << "destination_idx=" << destination_idx << endl
                //     << "union_=" << union_ << endl
                // )
                if (union_.size() > t.layers[col].activeRows.size()) {
                    assert(union_.size() == t.layers[col].activeRows.size() + 1);
                    assert(union_.back() == t.layers[col + 1].activeRows.back());
                    // I) a new row has been activated (exactly 1!!!)
                    // now there are two possible values of `w`
                    // II) a bit should be added to destination_idx. By
                    // construction of MSF, it's the last member of `union_`
                    for (u64 new_row_bit : {0, 1}) {
                        BitSpan(w).set(w.bits - 1, new_row_bit);
                        auto& node = t.layers[col].nodes[node_idx];
                        // debug(std::cout << "w=" << w << "\nx=" << x << "\nw^x=" << BitSpan::ScalarProduct(w, x) << endl)
                        node.to[(int)BitSpan::ScalarProduct(w, x)] =
                            destination_idx | BIT(t.layers[col + 1].activeRowToIdx[union_.back()]) * new_row_bit;
                    }
                } else {
                    // active rows are the same or even lesser
                    auto& node = t.layers[col].nodes[node_idx];
                    // debug(std::cout << "w=" << w << "\nx=" << x << "\nw^x=" << BitSpan::ScalarProduct(w, x) << endl)
                    node.to[BitSpan::ScalarProduct(w, x)] = destination_idx;
                }
            }

            // bits in `intersection`
        }

        return t;
    }

    // NOTE: out-parameter to make this zero-allocation
    void DecodeInplace(const std::vector<double>& y, BitVector& result) {
        // 1. Initialize trellis (from = NIL, metric = infty) TODO: infty or -infty?
        for (auto& layer : layers) {
            for (auto& node : layer.nodes) {
                node.metric_dp = INFTY;
            }
        }

        assert(layers[0].nodes.size() == 1);

        // 2. Do the DP
        layers[0].nodes[0].metric_dp = 0;
        for (u64 layer_idx = 0; layer_idx + 1 < layers.size(); layer_idx++) {
            auto& curLayer = layers[layer_idx];
            auto& nextLayer = layers[layer_idx + 1];

            for (u64 node_idx = 0; node_idx < curLayer.nodes.size(); node_idx++) {
                auto& node = curLayer.nodes[node_idx];
                for (int to : {0, 1}) {
                    if (node.to[to] == NIL) continue;

                    auto& son = nextLayer.nodes[node.to[to]];
                    auto difference = BIT_TO_SIGNAL[to] - y[layer_idx];
                    double suggestedMetric = node.metric_dp + std::abs(difference * difference);

                    if (suggestedMetric < son.metric_dp) {
                        son.metric_dp = suggestedMetric;
                        son.parent_idx = node_idx;
                        son.parent_label_dp = to;
                    }
                }
            }
        }

        // 3. Backtrack the answer
        for (u64 layer_idx = layers.size() - 1, node_idx = 0; layer_idx != 0; layer_idx--) {
            auto& node = layers[layer_idx].nodes[node_idx];
            BitSpan(result).set(layer_idx - 1, node.parent_label_dp);
            node_idx = node.parent_idx;
        }
    }

    [[nodiscard]] static Trellis FromGeneratorMatrix(const BitMatrix& m) {
        return FromMSF(m.GetMinimalSpanForm());
    }

    [[nodiscard]] std::vector<u64> GetComplexityProfile() const {
        std::vector<u64> profile;
        profile.reserve(layers.size());
        for (const auto& layer : layers) {
            profile.push_back(layer.nodes.size());
        }
        return profile;
    }


    const BitMatrix msf;
    std::vector<Layer> layers;
};

struct Solver {

    [[nodiscard]] static Solver FromGeneratorMatrix(const BitMatrix& g) {
        BitMatrix gT(/* rows = */ g.columns, /* columns = */ g.rows);
        for (u64 i = 0; i < g.rows; i++) {
            for (u64 j = 0; j < g.columns; j++) {
                gT.set(j, i, g.get(i, j));
            }
        }
        return {
            g,
            gT,
            Trellis::FromGeneratorMatrix(g),
        };
    }

    [[nodiscard]] double Simulate(const double noiseLevel, const u64 iterations, const u64 maxErrors) {
        // See Lecture 1 slide 11 (alpha = 1)
        // sigma^2 := N0 / 2
        // OSW := 10 log_10{Eb/N0}  -- Otnosheniye Signal Wum na bit
        //
        // 10^{OSW/10} = Eb / N0
        // N0 = Eb / 10^{OSW/10}
        // sigma^2 = 0.5 * Eb / 10 ^ {OSW/10}
        // Eb := Es / R = Es / (k / n)
        // Es := alpha^2 = 1
        // Eb = 1 / (k / n) = n / k
        //
        // OSW = noiseLevel
        // sigma^2 = 0.5 * (n/k) / 10 ^ {noiseLevel / 10}
        const double N = g.columns;
        const double K = g.rows;
        std::normal_distribution<double> noiser(0, std::sqrt(0.5 * N / K / std::pow(10, noiseLevel / 10)));
        std::bernoulli_distribution coin(0.5);

        BitVector generatedVector(K);
        BitVector encodedVector(N);
        BitVector decodedVector(N);
        std::vector<double> noisedVector(N);

        u64 errors = 0;
        u64 actualIterations = 0;

        for (actualIterations = 0; actualIterations < iterations && errors < maxErrors; actualIterations++) {
            for (u64 i = 0; i < generatedVector.bits; i++)
                BitSpan(generatedVector).set(i, coin(gen));

            EncodeInplace(generatedVector, encodedVector);

            for (u64 i = 0; i < encodedVector.bits; i++) {
                noisedVector[i] = BIT_TO_SIGNAL[BitSpan(encodedVector).get(i)] + noiser(gen);
            }

            trellis.DecodeInplace(noisedVector, decodedVector);

            errors += !(encodedVector == decodedVector);
        }

        debug(std::cout << "Simulate: errors=" << errors << " iterations=" << actualIterations << endl);
        return double(errors) / actualIterations;
    }

    [[nodiscard]] BitVector Decode(const std::vector<double>& y) {
        BitVector result(g.columns);
        trellis.DecodeInplace(y, result);
        return result;
    }

    [[nodiscard]] BitVector Encode(const BitVector& x) const {
        BitVector result{g.columns};
        EncodeInplace(x, result);
        return result;
    }

    /// Make less allocations in case of frequent encoding
    void EncodeInplace(const BitVector& x, BitVector& out) const {
        assert(x.bits == g.rows);
        // gT.rows == g.columns
        for (u64 i = 0; i < gT.rows; i++) {
            BitSpan(out).set(
                i,
                BitSpan::ScalarProduct(
                    const_cast<BitMatrix&>(gT).getRow(i),
                    BitSpan(const_cast<BitVector&>(x))
                )
            );
        }
    }


    BitMatrix g;
    BitMatrix gT;  // transposed version for faster Encode
    Trellis trellis;
    std::mt19937_64 gen;
};

#ifndef TEST
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    return 0;
}
#endif
