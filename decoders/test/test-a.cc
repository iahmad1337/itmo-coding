#define TEST

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <bitset>
#include <fstream>
#include <sstream>

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

TEST(TestBitwise, TestBitParity) {
    for (u64 i = 0; i < (1ull << 20); i++) {
        std::bitset<64> bs{i};
        ASSERT_EQ(bs.count() & 1, bit_parity(i));
    }
    for (u64 i = BIT(48); i < BIT(48) + BIT(20); i++) {
        std::bitset<64> bs{i};
        ASSERT_EQ(bs.count() & 1, bit_parity(i));
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

class TestScalarProduct : public testing::TestWithParam<std::tuple<std::string, std::string, int>> {};

using TSP = std::tuple<std::string, std::string, int>;

BitVector FromString(const std::string_view str) {
    u64 bits = 0;
    for (char c : str) {
        bits += c == '0' || c == '1';
    }
    BitVector result(bits);
    u64 cur = 0;
    for (char c : str) {
        bool value = false;
        if (c == '0') value = false;
        else if (c == '1') value = true;
        else continue;
        BitSpan(result).set(cur, value);
        cur++;
    }
    return result;
}

BitMatrix FromString(const std::string_view str, int rows, int columns) {
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

TEST_P(TestScalarProduct, Valid) {
    auto [x, y, result] = GetParam();

    auto xVec = FromString(x);
    auto yVec = FromString(y);

    ASSERT_EQ(BitSpan::ScalarProduct(xVec, yVec), result);
}

INSTANTIATE_TEST_SUITE_P(
    SomeName,
    TestScalarProduct,
    testing::Values(
        TSP{"101", "101", 0},
        TSP{"011", "111", 0},
        TSP{"111", "111", 1},
        TSP{"101", "010", 0},
        TSP{"101", "111", 0},
        TSP{"101", "110", 1}
    )
);

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
    {
        R"(
        1 1 1 0 1
        0 0 1 0 0
        0 1 1 0 1
        )",
        3, 5
    },
    {
        R"(
        10000001
        01000001
        00100001
        00010001
        00001001
        00000101
        00000011
        )",
        7, 8
    },
};

TEST(TestMatrix, TestMSF) {
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

[[nodiscard]] std::string TrellisToDot(const Trellis& t) {
    std::stringstream ss;

    // Header
    // To increase/decrease spacing between nodes, play around with `pad`,
    // `nodesep` and `ranksep`
    ss << R"(
        strict digraph {
            graph [
                rankdir = LR,
                pad = "0.5",
                nodesep = "0.25",
                ranksep = "1.5"
            ]
    )";

    // graph-level unique node
    constexpr auto toLabel = [](u64 layer, u64 node) {
        return "l" + std::to_string(layer) + "n" + std::to_string(node);
    };

    // layer-level unique node (may repeat between layers)
    constexpr auto toNodeLabel = [](u64 node, u64 activeRowsCount) {
        std::string result;
        result.resize(activeRowsCount, '0');
        for (u64 i = 0; node != 0; node >>= 1, i++) {
            if (node & 1)
                result[i] = '1';
        }
        if (activeRowsCount == 0) {
            result = "sentinel";
        }
        result.insert(result.begin(), '"');
        result.insert(result.end(), '"');
        return result;
    };

    u64 layer = 1;
    for (const auto& [nodes, activeRows, _] : t.layers) {
        ss << "subgraph Layer" << layer << " {\n"
            << "cluster=true;\n"
            << "peripheries=0;\n"
            << "label=\"" << activeRows << "\";\n"
            << "rankdir=BT;\n";

        // label nodes
        for (u64 node = 0; node < nodes.size(); node++) {
            ss << toLabel(layer, node) << " [label=" << toNodeLabel(node, activeRows.size());
            if (nodes[node].metric_dp != INFTY) ss << ", xlabel=" << nodes[node].metric_dp;
            ss << "];\n";
        }

        // subgraph footer
        ss << "}\n";
        layer++;
    }

    {
        // Beware - the alignment hack: https://stackoverflow.com/questions/49348639/graphviz-aligning-nodes-and-clusters
        layer = 1;
        for (; layer <= t.layers.size(); layer++) {
            if (layer != 1) ss << ", ";
            ss << toLabel(layer, 0);
        }
        ss << "[group=1];" << endl;
    }

    // [optional] make edges straight
    // ss << "splines=false" << endl;

    // draw edges
    layer = 1;
    for (const auto& [nodes, activeRows, _] : t.layers) {
        for (u64 node = 0; node < nodes.size(); node++) {
            const auto& n = nodes[node];
            if (n.to[0] != NIL)
                ss << toLabel(layer, node) << " -> " << toLabel(layer + 1, n.to[0]) << "[label=0,color=blue,fontcolor=blue];\n";
            if (n.to[1] != NIL)
                ss << toLabel(layer, node) << " -> " << toLabel(layer + 1, n.to[1]) << "[label=1,color=red,fontcolor=red];\n";
        }
        layer++;
    }

    // Footer
    ss << "}";
    return ss.str();
}

void DisplayTrellis(const Trellis& t) {
    auto dotContents = TrellisToDot(t);
    std::ofstream of("trellis.dot");
    of << dotContents << endl;
    of.close();

    system("dot -T png trellis.dot >trellis.png");
    system("open trellis.png || wslview trellis.png");
}

TEST(TestMatrix, TestTrellis) {
    constexpr auto getProfile = [] (arg m) {
        auto g = FromString(m.m, m.rows, m.columns);
        auto trellis = Trellis::FromGeneratorMatrix(g);
        // DisplayTrellis(trellis);
        return trellis.GetComplexityProfile();
    };

    using namespace testing;
    EXPECT_THAT(getProfile(ms[0]), ElementsAre(1, 2, 4, 8, 4, 8, 4, 2, 1));
    // TODO: calculate complexity profiles for other matrices
}

TEST(TestDecode, SampleDecodeTest) {
    // Hamming code
    Solver s = Solver::FromGeneratorMatrix(FromString(ms[0].m, ms[0].rows, ms[0].columns));

    // sample from the statements
    std::vector<double> y = {-1.0, 1.0, 1, 1, 1, 1, 1, 1.5};
    auto result = s.Decode(y);
    ASSERT_EQ(BitSpan(result).to_string(), "00000000");
    if (getenv("DISPLAY") != nullptr) {
        DisplayTrellis(s.trellis);
    }

    y = {2, 2, 2, 2, 2, 2, 2, 2};
    result = s.Decode(y);
    ASSERT_EQ(BitSpan(result).to_string(), "00000000");

    y = {-1, -1, -1, -1, -1, -1, -1, -1};
    result = s.Decode(y);
    ASSERT_EQ(BitSpan(result).to_string(), "11111111");

    y = {-1, -1, -1, -1, 1, 1, 1, 1};
    result = s.Decode(y);
    ASSERT_EQ(BitSpan(result).to_string(), "11110000");

    y = {1, 1, 1, 1, -1, -1, -1, -1};
    result = s.Decode(y);
    ASSERT_EQ(BitSpan(result).to_string(), "00001111");

    y = {1, -1, 1, -1, 1, -1, 1, -1};
    result = s.Decode(y);
    ASSERT_EQ(BitSpan(result).to_string(), "01010101");

    y = {-1, 1, -1, 1, 1, -1, 1, -1};
    result = s.Decode(y);
    ASSERT_EQ(BitSpan(result).to_string(), "10100101");
}

TEST(TestDecode, LectureSlidesDecodeTest) {
    // TODO
}

TEST(TestEncode, SampleEncodeTest) {
    Solver s = Solver::FromGeneratorMatrix(FromString(ms[0].m, ms[0].rows, ms[0].columns));

    BitVector x = FromString("1000");
    auto result = s.Encode(x);
    ASSERT_EQ(BitSpan(result).to_string(), "11111111");

    x = FromString("0100");
    result = s.Encode(x);
    ASSERT_EQ(BitSpan(result).to_string(), "11110000");

    x = FromString("0010");
    result = s.Encode(x);
    ASSERT_EQ(BitSpan(result).to_string(), "11001100");

    x = FromString("0001");
    result = s.Encode(x);
    ASSERT_EQ(BitSpan(result).to_string(), "10101010");

    x = FromString("1001");
    result = s.Encode(x);
    ASSERT_EQ(BitSpan(result).to_string(), "01010101");

    x = FromString("0000");
    result = s.Encode(x);
    ASSERT_EQ(BitSpan(result).to_string(), "00000000");
}

TEST(TestSimulate, SampleSimulateTest) {
    Solver s = Solver::FromGeneratorMatrix(FromString(ms[0].m, ms[0].rows, ms[0].columns));

    EXPECT_NEAR(s.Simulate(3, 100000, 100), 0.0256, /* abs_error= */ 0.005);
    EXPECT_NEAR(s.Simulate(4, 100000, 100), 0.00931, /* abs_error= */ 0.0025);
}
