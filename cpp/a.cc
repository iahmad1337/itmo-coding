#include <iostream>
#include <absl/strings/str_format.h>
#include <gtest/gtest.h>

int main([[maybe_unused]] int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  std::cout << absl::StrFormat("I was called with command %s", argv[0]) << std::endl;
  return RUN_ALL_TESTS();
}
