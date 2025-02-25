#include <iostream>
#include <vector>
#include <cassert>

#if defined(__x86_64__)
#include <x86intrin.h>
#endif

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

    bool get(u64 idx) const {
        assert(idx < bits);
        return (data[idx / 64] & (idx & 0x3f)) != 0;
    }

    /// Position of first set bit
    std::optional<u64> first_one() const {
        for (auto it = cbegin(); it != cend(); it++) {
            if (*it != 0) {
                return (it - cbegin()) * 64 + *bsf(*it);
            }
        }
        return std::nullopt;
    }

    /// Position of last set bit
    std::optional<u64> last_one() const {
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

    u64* begin() const {
        return data;
    }

    u64* end() const {
        return data + UPP(bits + 63);
    }

    const u64* cbegin() const {
        return const_cast<BitSpan*>(this)->begin();
    }

    const u64* cend() const {
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

    BitSpan shrink(u64 bits) {
        return BitSpan{data, bits};
    }

    static bool ScalarProduct(const BitSpan& lhs, const BitSpan& rhs) {
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

    operator BitSpan() {
        return BitSpan{storage.data(), bits};
    }

    std::vector<u64> storage;
    u64 bits;
};

struct BitMatrix {
    // TODO
};

#ifndef TEST
int main() {

    return 0;
}
#endif
