#include "json.h"

#include <charconv>
#include <cstdio>

namespace vredit {

namespace {

constexpr std::size_t kMaxJsonDepth = 256;

bool isDigit(char c) {
  return c >= '0' && c <= '9';
}

bool isHex(char c) {
  return isDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

std::uint32_t hexValue(char c) {
  if (isDigit(c)) {
    return static_cast<std::uint32_t>(c - '0');
  }
  if (c >= 'a' && c <= 'f') {
    return static_cast<std::uint32_t>(c - 'a') + 10U;
  }
  return static_cast<std::uint32_t>(c - 'A') + 10U;
}

void appendUtf8(std::string& out, std::uint32_t codePoint) {
  if (codePoint < 0x80U) {
    out.push_back(static_cast<char>(codePoint));
  } else if (codePoint < 0x800U) {
    out.push_back(static_cast<char>(0xC0U | (codePoint >> 6U)));
    out.push_back(static_cast<char>(0x80U | (codePoint & 0x3FU)));
  } else if (codePoint < 0x10000U) {
    out.push_back(static_cast<char>(0xE0U | (codePoint >> 12U)));
    out.push_back(static_cast<char>(0x80U | ((codePoint >> 6U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | (codePoint & 0x3FU)));
  } else {
    out.push_back(static_cast<char>(0xF0U | (codePoint >> 18U)));
    out.push_back(static_cast<char>(0x80U | ((codePoint >> 12U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | ((codePoint >> 6U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | (codePoint & 0x3FU)));
  }
}

} // namespace

struct JsonParser {
  std::string_view text;
  std::size_t pos{0};
  std::size_t depth{0};
  JsonDocument& document;
  std::vector<JsonDocument::Node>& nodes;
  std::string& error;

  JsonParser(std::string_view input, JsonDocument& doc, std::string& err)
      : text(input), document(doc), nodes(doc.nodes_), error(err) {}

  bool fail(char const* message) {
    char buffer[128];
    std::snprintf(buffer, sizeof(buffer), "json error at byte %zu: %s", pos, message);
    error = buffer;
    return false;
  }

  void skipWhitespace() {
    while (pos < text.size()) {
      char c = text[pos];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++pos;
      } else {
        break;
      }
    }
  }

  bool consume(char expected) {
    if (pos < text.size() && text[pos] == expected) {
      ++pos;
      return true;
    }
    return false;
  }

  bool parseString(std::string& out) {
    if (!consume('"')) {
      return fail("expected string");
    }
    out.clear();
    while (pos < text.size()) {
      char c = text[pos++];
      if (c == '"') {
        return true;
      }
      if (c == '\\') {
        if (pos >= text.size()) {
          return fail("unterminated escape");
        }
        char escape = text[pos++];
        switch (escape) {
        case '"':
          out.push_back('"');
          break;
        case '\\':
          out.push_back('\\');
          break;
        case '/':
          out.push_back('/');
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u': {
          if (pos + 4U > text.size()) {
            return fail("truncated \\u escape");
          }
          std::uint32_t code = 0;
          for (int i = 0; i < 4; ++i) {
            char hex = text[pos + static_cast<std::size_t>(i)];
            if (!isHex(hex)) {
              return fail("bad \\u escape");
            }
            code = (code << 4U) | hexValue(hex);
          }
          pos += 4U;
          if (code >= 0xD800U && code <= 0xDBFFU) {
            // High surrogate: expect \uXXXX low surrogate.
            if (pos + 6U > text.size() || text[pos] != '\\' || text[pos + 1U] != 'u') {
              return fail("unpaired surrogate");
            }
            std::uint32_t low = 0;
            for (int i = 0; i < 4; ++i) {
              char hex = text[pos + 2U + static_cast<std::size_t>(i)];
              if (!isHex(hex)) {
                return fail("bad \\u escape");
              }
              low = (low << 4U) | hexValue(hex);
            }
            if (low < 0xDC00U || low > 0xDFFFU) {
              return fail("unpaired surrogate");
            }
            pos += 6U;
            code = 0x10000U + ((code - 0xD800U) << 10U) + (low - 0xDC00U);
          } else if (code >= 0xDC00U && code <= 0xDFFFU) {
            return fail("unpaired surrogate");
          }
          appendUtf8(out, code);
          break;
        }
        default:
          return fail("unknown escape");
        }
      } else {
        out.push_back(c);
      }
    }
    return fail("unterminated string");
  }

  bool parseValue(JsonDocument::NodeId& id) {
    if (depth >= kMaxJsonDepth) {
      return fail("nesting too deep");
    }
    ++depth;
    bool ok = parseValueInner(id);
    --depth;
    return ok;
  }

  bool parseValueInner(JsonDocument::NodeId& id) {
    skipWhitespace();
    if (pos >= text.size()) {
      return fail("unexpected end of input");
    }
    char c = text[pos];
    JsonDocument::NodeId newIndex = static_cast<JsonDocument::NodeId>(nodes.size());
    nodes.push_back(JsonDocument::Node{});
    JsonDocument::Node& node = nodes[newIndex];
    id = newIndex;

    if (c == '{') {
      ++pos;
      node.type = JsonDocument::Type::Object;
      skipWhitespace();
      if (consume('}')) {
        return true;
      }
      while (true) {
        skipWhitespace();
        std::string key;
        if (!parseString(key)) {
          return false;
        }
        skipWhitespace();
        if (!consume(':')) {
          return fail("expected ':'");
        }
        JsonDocument::NodeId value = JsonDocument::kInvalid;
        if (!parseValue(value)) {
          return false;
        }
        JsonDocument::Node& objectNode = nodes[newIndex]; // parseValue may reallocate
        objectNode.keys.push_back(key);
        objectNode.children.push_back(value);
        skipWhitespace();
        if (consume(',')) {
          continue;
        }
        if (consume('}')) {
          return true;
        }
        return fail("expected ',' or '}'");
      }
    }

    if (c == '[') {
      ++pos;
      node.type = JsonDocument::Type::Array;
      skipWhitespace();
      if (consume(']')) {
        return true;
      }
      while (true) {
        JsonDocument::NodeId value = JsonDocument::kInvalid;
        if (!parseValue(value)) {
          return false;
        }
        nodes[newIndex].children.push_back(value);
        skipWhitespace();
        if (consume(',')) {
          continue;
        }
        if (consume(']')) {
          return true;
        }
        return fail("expected ',' or ']'");
      }
    }

    if (c == '"') {
      node.type = JsonDocument::Type::String;
      return parseString(node.string);
    }

    if (text.substr(pos, 4) == "true") {
      pos += 4U;
      node.type = JsonDocument::Type::Bool;
      node.boolean = true;
      return true;
    }
    if (text.substr(pos, 5) == "false") {
      pos += 5U;
      node.type = JsonDocument::Type::Bool;
      node.boolean = false;
      return true;
    }
    if (text.substr(pos, 4) == "null") {
      pos += 4U;
      node.type = JsonDocument::Type::Null;
      return true;
    }

    // Number: scan the token, then convert strictly.
    std::size_t start = pos;
    if (consume('-')) {
      // sign consumed
    }
    while (pos < text.size() && (isDigit(text[pos]) || text[pos] == '.' || text[pos] == 'e' ||
                                 text[pos] == 'E' || text[pos] == '+' || text[pos] == '-')) {
      ++pos;
    }
    if (pos == start) {
      return fail("unexpected character");
    }
    double value = 0.0;
    const char* first = text.data() + start;
    const char* last = text.data() + pos;
    auto result = std::from_chars(first, last, value);
    if (result.ec != std::errc{} || result.ptr != last) {
      return fail("bad number");
    }
    node.type = JsonDocument::Type::Number;
    node.number = value;
    return true;
  }

  bool run() {
    JsonDocument::NodeId rootId = JsonDocument::kInvalid;
    if (!parseValue(rootId)) {
      return false;
    }
    skipWhitespace();
    if (pos != text.size()) {
      return fail("trailing content");
    }
    document.root_ = rootId;
    return true;
  }
};

bool JsonDocument::parse(std::string_view text, std::string& error) {
  nodes_.clear();
  root_ = kInvalid;
  error.clear();
  JsonParser parser(text, *this, error);
  return parser.run();
}

JsonDocument::NodeId JsonDocument::root() const {
  return root_;
}

JsonDocument::Type JsonDocument::type(NodeId id) const {
  return id < nodes_.size() ? nodes_[id].type : Type::Null;
}

bool JsonDocument::isNull(NodeId id) const {
  return type(id) == Type::Null;
}

bool JsonDocument::asBool(NodeId id) const {
  return id < nodes_.size() && nodes_[id].type == Type::Bool ? nodes_[id].boolean : false;
}

double JsonDocument::asNumber(NodeId id) const {
  return id < nodes_.size() && nodes_[id].type == Type::Number ? nodes_[id].number : 0.0;
}

std::string const& JsonDocument::asString(NodeId id) const {
  static std::string const empty;
  return id < nodes_.size() && nodes_[id].type == Type::String ? nodes_[id].string : empty;
}

std::size_t JsonDocument::size(NodeId id) const {
  return id < nodes_.size() ? nodes_[id].children.size() : 0U;
}

JsonDocument::NodeId JsonDocument::element(NodeId array, std::size_t index) const {
  if (array >= nodes_.size() || index >= nodes_[array].children.size()) {
    return kInvalid;
  }
  return nodes_[array].children[index];
}

JsonDocument::NodeId JsonDocument::member(NodeId object, std::string_view key) const {
  if (object >= nodes_.size()) {
    return kInvalid;
  }
  Node const& node = nodes_[object];
  for (std::size_t i = 0; i < node.keys.size(); ++i) {
    if (node.keys[i] == key) {
      return node.children[i];
    }
  }
  return kInvalid;
}

std::string const& JsonDocument::memberKey(NodeId object, std::size_t index) const {
  static std::string const empty;
  if (object >= nodes_.size() || index >= nodes_[object].keys.size()) {
    return empty;
  }
  return nodes_[object].keys[index];
}

} // namespace vredit
