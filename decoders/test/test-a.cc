#define TEST

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <bitset>

#include "../src/a.cc"

TEST(TestBitwise, TestBsf) {
    EXPECT_EQ(bsf(0), std::nullopt);
    EXPECT_EQ(bsf(1), 0);
    EXPECT_EQ(bsf(2), 1);
    for (int i = 0; i < 63; i++) {
        EXPECT_EQ(bsf(BIT(i)), i);
    }
}

TEST(TestBitwise, TestBsr) {
    for (int i = 0; i < 63; i++) {
        EXPECT_EQ(bsf(BIT(i)), bsr(BIT(i)));
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

BitMatrix FromString(std::string str, int rows, int columns) {
    BitMatrix result(rows, columns);
    u64 cur = 0;
    for (char c : str) {
        bool value = false;
        if (c == '0') value = false;
        else if (c == '1') value = true;
        else continue;
        auto i = cur / columns;
        auto j = cur % columns;
        result.set(i, j, value);
        cur++;
    }
    return result;
}

std::string ToString(const BitMatrix& m) {
    std::stringstream ss;
    for (u64 row = 0; row < m.rows; row++) {
        for (u64 col = 0; col < m.columns; col++) {
            if (col != 0) ss << " ";
            ss << int(m.get(row, col));
        }
        ss << "\n";
    }
    return ss.str();
}

TEST(TestMatrix, TestSwap) {
    struct arg {
        std::string m;
        int rows;
        int columns;
    };
    arg ms[] = {
        {
            R"(
            11111111
            11110000
            11001100
            10101010
            )",
            4, 8
        },
        {
            R"(
            11110
            01101
            11100
            )",
            3, 5
        },
        {
            R"(
            11110110
            01101101
            10100100
            01001001
            )",
            4, 8
        },
    };
    for (const auto& m : ms) {
        auto g = FromString(m.m, m.rows, m.columns);
        auto msf = g.GetMinimalSpanForm();
        std::cout << "g=\n" << ToString(g) << std::endl;
        std::cout << "msf(g)=\n" << ToString(msf) << std::endl;

        // forall i: begin(i) < begin(i + 1)
        for (int i = 0; i + 1 < g.rows; i++) {
            EXPECT_LT(msf.getRow(i).first_one(), msf.getRow(i + 1).first_one());
        }

        // forall i != j: end(i) != end(j)
        for (int i = 0; i < g.rows; i++) {
            for (int j = 0; j < g.rows; j++) {
                if (i != j) {
                    EXPECT_NE(msf.getRow(i).last_one(), msf.getRow(j).last_one());
                }
            }
        }
    }

}
