#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vredit {

// Minimal strict JSON DOM, arena-based (nodes reference each other by index) so
// the type stays self-contained and the compiler stays in charge. Sufficient
// for glTF 2.0 metadata: objects, arrays, strings (with escapes), numbers,
// booleans and null.
class JsonDocument {
public:
  enum class Type : std::uint8_t { Null, Bool, Number, String, Array, Object };
  using NodeId = std::uint32_t;

  static constexpr NodeId kInvalid = UINT32_MAX;

  // Parses `text`, replacing any previous content. Returns false with a
  // human-readable error (with byte offset) on any malformed input or trailing
  // garbage.
  bool parse(std::string_view text, std::string& error);

  NodeId root() const;
  Type type(NodeId id) const;
  bool isNull(NodeId id) const;
  bool asBool(NodeId id) const;
  double asNumber(NodeId id) const;
  std::string const& asString(NodeId id) const;

  std::size_t size(NodeId id) const;                             // array/object member count
  NodeId element(NodeId array, std::size_t index) const;         // array element
  NodeId member(NodeId object, std::string_view key) const;      // object member; kInvalid if absent
  std::string const& memberKey(NodeId object, std::size_t index) const;

private:
  struct Node {
    Type type{Type::Null};
    bool boolean{false};
    double number{0.0};
    std::string string;
    std::vector<NodeId> children;
    std::vector<std::string> keys; // object keys, parallel to children
  };

  friend struct JsonParser;

  std::vector<Node> nodes_;
  NodeId root_{kInvalid};
};

} // namespace vredit
