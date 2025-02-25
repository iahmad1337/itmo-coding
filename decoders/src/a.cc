#include <vector>
#include <cassert>

// TODO: using vector = small_vector<u64, 64>; // 64 bytes inplace-storage

using i32 = int32_t;
using i64 = int64_t;
using u32 = uint32_t;
using u64 = uint64_t;

#define SUB(i) ((i) & 0x3f)
#define UPP(i) ((i) / 64)
#define BIT(i) (1ull << (i))

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
                // TODO: invalid
                return ((it - cbegin()) * 64 + *bsr(*it));
            }
        } while (it != cbegin());
        return std::nullopt;
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

    [[nodiscard]] BitSpan shrinked(u64 bits) {
        return BitSpan{data, bits};
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
        return popcnt(result);
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
            if (row_with_one != rows) {
                assert(row <= row_with_one && row_with_one < rows);
                this->swap(row, row_with_one);
            }

            for (u64 lower_row = row + 1; lower_row < rows; lower_row++) {
                if (get(lower_row, col)) {
                    getRow(lower_row) ^= getRow(row);
                    assert(!get(lower_row, col));
                }
            }
        }  // Post: forall i: begin(i) < begin(i + 1)

        for (u64 row = rows - 1; row != -1; row--) {
            // Invariant:
            auto last_one = getRow(row).last_one();
            assert (bool(last_one));
            for (u64 upper_row = 0; upper_row < row; upper_row++) {
                getRow(upper_row) ^= getRow(row);
            }  // Post: forall i != row: G[i][end(row)] == 0
        }  // Post: forall i != j: end(i) != end(j)
    }

    const u64 rows;
    const u64 columns;
    const u64 stride;  // number of limbs per row
    std::vector<u64> storage;
};

#ifndef TEST
int main() {

    return 0;
}
#endif
