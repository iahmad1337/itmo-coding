#define TEST

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <bitset>

#include "../src/a.cc"

#define P2(i) (1ull << (i))

TEST(TestBitwise, TestBsf) {
    EXPECT_EQ(bsf(0), std::nullopt);
    EXPECT_EQ(bsf(1), 0);
    EXPECT_EQ(bsf(2), 1);
    for (int i = 0; i < 63; i++) {
        EXPECT_EQ(bsf(P2(i)), i);
    }
}

TEST(TestBitwise, TestBsr) {
    for (int i = 0; i < 63; i++) {
        EXPECT_EQ(bsf(P2(i)), bsr(P2(i)));
    }
}

TEST(TestBitwise, TestPopcnt) {
    for (u64 i = 0; i < 256*256*256 + 1; i++) {
        std::bitset<64> bs{i};
        ASSERT_EQ(bs.count(), popcnt(i));
    }
    for (u64 i = BIT(48); i < BIT(48) + 256*256 + 1; i++) {
        std::bitset<64> bs{i};
        ASSERT_EQ(bs.count(), popcnt(i));
    }
}

TEST(TestVector, TestBitSpan) {
    auto v = BitVector(255);

    auto span = BitSpan(v);
    EXPECT_EQ(span.first_one(), std::nullopt);
    EXPECT_EQ(span.last_one(), std::nullopt);

    span.set(0, 1);
    EXPECT_EQ(span.first_one(), span.last_one());
    EXPECT_EQ(span.first_one(), 0);

    span.set(128, 1);
    EXPECT_EQ(span.first_one(), 0);
    EXPECT_EQ(span.last_one(), 128);

    span.set(254, 1);
    EXPECT_EQ(span.first_one(), 0);
    EXPECT_EQ(span.last_one(), 254);

    span.set(0, 0);
    EXPECT_EQ(span.first_one(), 128);
    EXPECT_EQ(span.last_one(), 254);

    span.set(254, 0);
    EXPECT_EQ(span.first_one(), 128);
    EXPECT_EQ(span.last_one(), 128);

    span.set(200, true);
    EXPECT_EQ(span.first_one(), 128);
    EXPECT_EQ(span.last_one(), 200);
}
