#include "obj.h"

#include <charconv>
#include <cmath>
#include <fstream>
#include <sstream>

namespace vredit {

namespace {

bool parseFloats(std::string_view line, std::vector<float>& out) {
  out.clear();
  std::size_t pos = 0;
  while (pos < line.size()) {
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) {
      ++pos;
    }
    if (pos >= line.size()) {
      break;
    }
    std::size_t start = pos;
    while (pos < line.size() && line[pos] != ' ' && line[pos] != '\t') {
      ++pos;
    }
    float value = 0.0F;
    const char* first = line.data() + start;
    const char* last = line.data() + pos;
    auto result = std::from_chars(first, last, value);
    if (result.ec != std::errc{} || result.ptr != last) {
      return false;
    }
    out.push_back(value);
  }
  return true;
}

bool parseIndexToken(std::string_view token, std::size_t count, std::int64_t& indexOut) {
  // Formats: "v", "v/vt", "v//vn", "v/vt/vn". The first field is the position
  // index; the third (if present) is the normal index. Strict: the token must
  // be exactly one integer (1-based; negatives count from the end; 0 is invalid).
  std::int64_t value = 0;
  const char* first = token.data();
  const char* last = token.data() + token.size();
  auto result = std::from_chars(first, last, value);
  if (result.ec != std::errc{} || result.ptr != last || value == 0) {
    return false;
  }
  if (value < 0) {
    indexOut = static_cast<std::int64_t>(count) + value;
  } else {
    indexOut = value - 1;
  }
  return indexOut >= 0;
}

std::string_view trimCarriage(std::string_view line) {
  if (!line.empty() && line.back() == '\r') {
    line.remove_suffix(1);
  }
  return line;
}

bool parseFaceVertex(std::string_view token, std::size_t positionCount, std::size_t normalCount,
                     std::int64_t& positionIndex, std::int64_t& normalIndex) {
  normalIndex = -1;
  std::size_t firstSlash = token.find('/');
  std::string_view positionPart =
      firstSlash == std::string_view::npos ? token : token.substr(0, firstSlash);
  if (!parseIndexToken(positionPart, positionCount, positionIndex)) {
    return false;
  }
  if (firstSlash == std::string_view::npos) {
    return true;
  }
  std::size_t secondSlash = token.find('/', firstSlash + 1U);
  if (secondSlash != std::string_view::npos && secondSlash + 1U < token.size()) {
    std::string_view normalPart = token.substr(secondSlash + 1U);
    return parseIndexToken(normalPart, normalCount, normalIndex);
  }
  return true; // "v/vt" or "v/vt/" — no normal
}
Vec3 lookupColour(MtlColours const& materials, std::string const& name) {
  for (auto const& [key, colour] : materials) {
    if (key == name) {
      return colour;
    }
  }
  return {1.0F, 1.0F, 1.0F};
}

} // namespace

bool parseMtl(std::string_view text, MtlColours& out) {
  out.clear();
  std::string currentName;
  std::size_t pos = 0;
  while (pos <= text.size()) {
    std::size_t end = text.find('\n', pos);
    if (end == std::string_view::npos) {
      end = text.size();
    }
    std::string_view line = trimCarriage(text.substr(pos, end - pos));
    pos = end + 1U;

    std::size_t lead = line.find_first_not_of(" \t");
    if (lead == std::string_view::npos) {
      continue;
    }
    std::size_t split = line.find_first_of(" \t", lead);
    std::string_view keyword =
        split == std::string_view::npos ? line.substr(lead) : line.substr(lead, split - lead);
    std::string_view rest = split == std::string_view::npos ? std::string_view{} : line.substr(split);

    if (keyword == "newmtl") {
      std::istringstream stream{std::string(rest)};
      stream >> currentName;
    } else if (keyword == "Kd" && !currentName.empty()) {
      std::vector<float> values;
      if (parseFloats(rest, values) && values.size() >= 3U) {
        out.push_back({currentName, {values[0], values[1], values[2]}});
      }
    }
    if (pos > text.size()) {
      break;
    }
  }
  return true;
}

namespace {

bool loadObjImpl(std::string_view text, MtlColours const& materials, Model& out,
                 std::string& error) {
  out = Model{};
  std::vector<Vec3> positions;
  std::vector<Vec3> normals;

  std::int32_t currentNode = -1;
  std::string currentGroupName;
  Vec3 currentColour{1.0F, 1.0F, 1.0F};
  std::size_t lineNo = 0;
  std::size_t pos = 0;

  auto fail = [&error](std::size_t line, char const* message) {
    error = "obj error on line " + std::to_string(line) + ": " + message;
    return false;
  };

  while (pos <= text.size()) {
    std::size_t end = text.find('\n', pos);
    if (end == std::string_view::npos) {
      end = text.size();
    }
    std::string_view line = trimCarriage(text.substr(pos, end - pos));
    pos = end + 1U;
    ++lineNo;

    // Split into first token and the rest, tolerating leading whitespace.
    std::size_t lead = line.find_first_not_of(" \t");
    std::size_t split = lead == std::string_view::npos ? std::string_view::npos
                                                       : line.find_first_of(" \t", lead);
    std::string_view keyword =
        lead == std::string_view::npos
            ? std::string_view{}
            : (split == std::string_view::npos ? line.substr(lead) : line.substr(lead, split - lead));
    std::string_view rest = split == std::string_view::npos ? std::string_view{} : line.substr(split);

    if (keyword.empty() || keyword == "#") {
      // blank or comment
    } else if (keyword == "v") {
      std::vector<float> values;
      if (!parseFloats(rest, values) || values.size() < 3U) {
        return fail(lineNo, "bad vertex");
      }
      positions.push_back({values[0], values[1], values[2]});
    } else if (keyword == "vn") {
      std::vector<float> values;
      if (!parseFloats(rest, values) || values.size() < 3U) {
        return fail(lineNo, "bad normal");
      }
      normals.push_back({values[0], values[1], values[2]});
    } else if (keyword == "vt" || keyword == "s" || keyword == "mtllib") {
      // No effect on geometry (mtllib is handled by loadObjFile).
    } else if (keyword == "usemtl") {
      std::istringstream stream{std::string(rest)};
      std::string name;
      stream >> name;
      if (currentNode >= 0) {
        ModelNode& node = out.nodes[static_cast<std::size_t>(currentNode)];
        if (node.mesh >= 0 &&
            !out.meshes[static_cast<std::size_t>(node.mesh)].data.vertices.empty()) {
          // Material change mid-group: geometry after this point belongs to a
          // new node so colours stay exact.
          currentNode = -1;
        }
      }
      currentColour = lookupColour(materials, name);
    } else if (keyword == "o" || keyword == "g") {
      std::istringstream stream{std::string(rest)};
      std::string name;
      stream >> name;
      currentGroupName =
          name.empty() ? std::string("group") + std::to_string(out.nodes.size()) : name;
      ModelNode node;
      node.name = currentGroupName;
      node.mesh = -1;
      out.nodes.push_back(std::move(node));
      currentNode = static_cast<std::int32_t>(out.nodes.size() - 1U);
      out.roots.push_back(currentNode);
    } else if (keyword == "f") {
      // Gather face vertices (any polygon), fan-triangulate.
      std::vector<std::int64_t> facePositions;
      std::vector<std::int64_t> faceNormals;
      std::size_t tokenStart = 0;
      while (tokenStart < rest.size()) {
        while (tokenStart < rest.size() && (rest[tokenStart] == ' ' || rest[tokenStart] == '\t')) {
          ++tokenStart;
        }
        if (tokenStart >= rest.size()) {
          break;
        }
        std::size_t tokenEnd = tokenStart;
        while (tokenEnd < rest.size() && rest[tokenEnd] != ' ' && rest[tokenEnd] != '\t') {
          ++tokenEnd;
        }
        std::int64_t positionIndex = -1;
        std::int64_t normalIndex = -1;
        if (!parseFaceVertex(rest.substr(tokenStart, tokenEnd - tokenStart), positions.size(),
                             normals.size(), positionIndex, normalIndex)) {
          return fail(lineNo, "bad face token");
        }
        if (positionIndex < 0 ||
            static_cast<std::size_t>(positionIndex) >= positions.size()) {
          return fail(lineNo, "face position index out of range");
        }
        if (normalIndex >= 0 && static_cast<std::size_t>(normalIndex) >= normals.size()) {
          return fail(lineNo, "face normal index out of range");
        }
        facePositions.push_back(positionIndex);
        faceNormals.push_back(normalIndex);
        tokenStart = tokenEnd;
      }
      if (facePositions.size() < 3U) {
        return fail(lineNo, "face with fewer than three vertices");
      }

      if (currentNode < 0) {
        ModelNode node;
        node.name = currentGroupName.empty() ? "default" : currentGroupName;
        node.mesh = -1;
        out.nodes.push_back(std::move(node));
        currentNode = static_cast<std::int32_t>(out.nodes.size() - 1U);
        out.roots.push_back(currentNode);
      }
      ModelNode& node = out.nodes[static_cast<std::size_t>(currentNode)];
      if (node.mesh < 0) {
        ModelMesh mesh;
        mesh.colour = currentColour;
        out.meshes.push_back(std::move(mesh));
        node.mesh = static_cast<std::int32_t>(out.meshes.size() - 1U);
      } else {
        out.meshes[static_cast<std::size_t>(node.mesh)].colour = currentColour;
      }
      MeshData& data = out.meshes[static_cast<std::size_t>(node.mesh)].data;

      for (std::size_t t = 1; t + 1U < facePositions.size(); ++t) {
        std::size_t const corners[3] = {0U, t, t + 1U};
        Vec3 const cornerPositions[3] = {
            positions[static_cast<std::size_t>(facePositions[corners[0]])],
            positions[static_cast<std::size_t>(facePositions[corners[1]])],
            positions[static_cast<std::size_t>(facePositions[corners[2]])]};
        Vec3 faceNormal = cross(cornerPositions[1] - cornerPositions[0],
                               cornerPositions[2] - cornerPositions[0]);
        faceNormal = length(faceNormal) > 1e-8F ? normalized(faceNormal) : Vec3{0.0F, 1.0F, 0.0F};
        for (std::size_t c = 0; c < 3U; ++c) {
          std::int64_t normalIndex = faceNormals[corners[c]];
          MeshVertex vertex;
          vertex.pos = cornerPositions[c];
          vertex.normal = normalIndex >= 0
                              ? normals[static_cast<std::size_t>(normalIndex)]
                              : faceNormal;
          vertex.colour = {1.0F, 1.0F, 1.0F};
          data.indices.push_back(static_cast<std::uint32_t>(data.vertices.size()));
          data.vertices.push_back(vertex);
        }
      }
    } else {
      // Tolerate unknown keywords (vp, l, p, vendor extensions ...): they carry
      // no geometry we consume.
    }

    if (pos > text.size()) {
      break;
    }
  }
  return true;
}

} // namespace

bool loadObj(std::string_view text, MtlColours const& materials, Model& out, std::string& error) {
  out = Model{};
  if (!loadObjImpl(text, materials, out, error)) {
    out = Model{}; // defined state on failure
    return false;
  }
  return true;
}

bool loadObjFile(char const* path, Model& out, std::string& error) {
  std::ifstream file(path);
  if (!file) {
    error = std::string("cannot open ") + path;
    return false;
  }
  std::ostringstream content;
  content << file.rdbuf();
  std::string text = content.str();
  if (text.empty()) {
    error = std::string("empty or unreadable ") + path;
    return false;
  }

  // Look for `mtllib` and load the sibling .mtl for usemtl colours.
  MtlColours materials;
  std::istringstream scan(text);
  std::string line;
  std::string directory(path);
  std::size_t slash = directory.find_last_of('/');
  directory = slash == std::string::npos ? std::string{} : directory.substr(0, slash + 1U);
  while (std::getline(scan, line)) {
    std::istringstream lineStream(line);
    std::string keyword;
    lineStream >> keyword;
    if (keyword == "mtllib") {
      std::string mtlName;
      lineStream >> mtlName;
      std::ifstream mtlFile(directory + mtlName);
      if (mtlFile) {
        std::ostringstream mtlContent;
        mtlContent << mtlFile.rdbuf();
        parseMtl(mtlContent.str(), materials);
      }
      break;
    }
  }

  return loadObj(text, materials, out, error);
}

} // namespace vredit
