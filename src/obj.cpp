#include "obj.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>

#include "image.h"

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
  // Strict: the token must be exactly one integer (1-based; negatives count
  // from the end; 0 is invalid).
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

// Parses a face corner token "v", "v/vt", "v//vn" or "v/vt/vn". Absent fields
// resolve to -1; resolved fields are range-checked.
bool parseFaceVertex(std::string_view token, std::size_t positionCount, std::size_t texcoordCount,
                     std::size_t normalCount, std::int64_t& positionIndex,
                     std::int64_t& texcoordIndex, std::int64_t& normalIndex) {
  texcoordIndex = -1;
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
  std::string_view middle =
      secondSlash == std::string_view::npos ? token.substr(firstSlash + 1U)
                                            : token.substr(firstSlash + 1U,
                                                           secondSlash - firstSlash - 1U);
  if (!middle.empty()) {
    if (!parseIndexToken(middle, texcoordCount, texcoordIndex) || texcoordIndex < 0) {
      return false;
    }
  }
  if (secondSlash != std::string_view::npos && secondSlash + 1U < token.size()) {
    return parseIndexToken(token.substr(secondSlash + 1U), normalCount, normalIndex) &&
           normalIndex >= 0;
  }
  return true; // "v/vt" or "v/vt/" — no normal
}

MtlMaterial const* lookupMaterial(MtlMaterials const& materials, std::string const& name) {
  for (MtlMaterial const& material : materials) {
    if (material.name == name) {
      return &material;
    }
  }
  return nullptr;
}

bool sameMaterial(MtlMaterial const& a, MtlMaterial const& b) {
  return a.colour.x == b.colour.x && a.colour.y == b.colour.y && a.colour.z == b.colour.z &&
         a.dissolve == b.dissolve && a.texture == b.texture &&
         a.normalTexture == b.normalTexture && a.normalScale == b.normalScale;
}

// MTL map statements may carry option flags (-s u v w, -o u v w, -bm value,
// ...). Flag arities vary, so the path is the LAST token (paths with unquoted
// spaces are not supported). A -bm value is captured when requested.
std::string parseMapStatement(std::string_view rest, float* bumpScaleOut) {
  std::istringstream stream{std::string(rest)};
  std::string path;
  std::string token;
  std::string previous;
  while (stream >> token) {
    if (previous == "-bm" && bumpScaleOut != nullptr) {
      std::istringstream value{token};
      float scale = 1.0F;
      if (value >> scale) {
        *bumpScaleOut = scale;
      }
    }
    previous = token;
    if (!token.empty() && token[0] == '-') {
      continue;
    }
    path = token;
  }
  return path;
}

} // namespace

bool parseMtl(std::string_view text, MtlMaterials& out) {
  out.clear();
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
      MtlMaterial material;
      stream >> material.name;
      out.push_back(std::move(material));
    } else if (!out.empty()) {
      MtlMaterial& material = out.back();
      std::vector<float> values;
      if (keyword == "Kd" && parseFloats(rest, values) && values.size() >= 3U) {
        material.colour = {values[0], values[1], values[2]};
      } else if (keyword == "d" && parseFloats(rest, values) && !values.empty()) {
        material.dissolve = values[0];
      } else if (keyword == "Tr" && parseFloats(rest, values) && !values.empty()) {
        material.dissolve = 1.0F - values[0];
      } else if (keyword == "map_Kd") {
        std::string path = parseMapStatement(rest, nullptr);
        if (!path.empty()) {
          material.mapKd = path;
        }
      } else if (keyword == "map_Bump" || keyword == "bump" || keyword == "norm" ||
                 keyword == "map_Kn") {
        std::string path = parseMapStatement(rest, &material.normalScale);
        if (!path.empty()) {
          material.mapBump = path;
        }
      }
    }
    if (pos > text.size()) {
      break;
    }
  }
  return true;
}

namespace {

bool loadObjImpl(std::string_view text, MtlMaterials const& materials,
                 std::vector<ModelTexture> const& textures, Model& out, std::string& error) {
  out = Model{};
  out.textures = textures;
  std::vector<Vec3> positions;
  std::vector<Vec3> normals;
  std::vector<Vec2> texcoords;

  std::int32_t currentNode = -1;
  std::string currentGroupName;
  MtlMaterial currentMaterial;
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
    } else if (keyword == "vt") {
      std::vector<float> values;
      if (!parseFloats(rest, values) || values.size() < 2U) {
        return fail(lineNo, "bad texcoord");
      }
      texcoords.push_back({values[0], values[1]});
    } else if (keyword == "s" || keyword == "mtllib") {
      // No effect on geometry (mtllib is handled by loadObjFile).
    } else if (keyword == "usemtl") {
      std::istringstream stream{std::string(rest)};
      std::string name;
      stream >> name;
      MtlMaterial found;
      if (MtlMaterial const* material = lookupMaterial(materials, name)) {
        found = *material;
      }
      if (currentNode >= 0 && !sameMaterial(currentMaterial, found)) {
        ModelNode& node = out.nodes[static_cast<std::size_t>(currentNode)];
        if (node.mesh >= 0 &&
            !out.meshes[static_cast<std::size_t>(node.mesh)].data.vertices.empty()) {
          // Material change mid-group: geometry after this point belongs to a
          // new node so colours stay exact.
          currentNode = -1;
        }
      }
      currentMaterial = found;
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
      // Gather face corners (any polygon), fan-triangulate.
      std::vector<std::int64_t> facePositions;
      std::vector<std::int64_t> faceTexcoords;
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
        std::int64_t texcoordIndex = -1;
        std::int64_t normalIndex = -1;
        if (!parseFaceVertex(rest.substr(tokenStart, tokenEnd - tokenStart), positions.size(),
                             texcoords.size(), normals.size(), positionIndex, texcoordIndex,
                             normalIndex)) {
          return fail(lineNo, "bad face token");
        }
        if (positionIndex < 0 ||
            static_cast<std::size_t>(positionIndex) >= positions.size()) {
          return fail(lineNo, "face position index out of range");
        }
        if (texcoordIndex >= 0 &&
            static_cast<std::size_t>(texcoordIndex) >= texcoords.size()) {
          return fail(lineNo, "face texcoord index out of range");
        }
        if (normalIndex >= 0 && static_cast<std::size_t>(normalIndex) >= normals.size()) {
          return fail(lineNo, "face normal index out of range");
        }
        facePositions.push_back(positionIndex);
        faceTexcoords.push_back(texcoordIndex);
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
        out.meshes.push_back(std::move(mesh));
        node.mesh = static_cast<std::int32_t>(out.meshes.size() - 1U);
      }
      ModelMesh& mesh = out.meshes[static_cast<std::size_t>(node.mesh)];
      mesh.colour = currentMaterial.colour;
      mesh.alphaFactor = currentMaterial.dissolve;
      mesh.texture = currentMaterial.texture;
      mesh.normalTexture = currentMaterial.normalTexture;
      mesh.normalScale = currentMaterial.normalScale;
      mesh.alpha = currentMaterial.dissolve < 1.0F ? AlphaMode::Blend : AlphaMode::Opaque;
      MeshData& data = mesh.data;

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
          std::int64_t texcoordIndex = faceTexcoords[corners[c]];
          MeshVertex vertex;
          vertex.pos = cornerPositions[c];
          vertex.normal = normalIndex >= 0
                              ? normals[static_cast<std::size_t>(normalIndex)]
                              : faceNormal;
          vertex.colour = {1.0F, 1.0F, 1.0F};
          vertex.uv = texcoordIndex >= 0 ? texcoords[static_cast<std::size_t>(texcoordIndex)]
                                         : Vec2{0.0F, 0.0F};
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

bool loadObj(std::string_view text, MtlMaterials const& materials,
             std::vector<ModelTexture> const& textures, Model& out, std::string& error) {
  out = Model{};
  if (!loadObjImpl(text, materials, textures, out, error)) {
    out = Model{}; // defined state on failure
    return false;
  }
  return true;
}

bool loadObjFile(char const* path, Model& out, std::string& error, bool withImages) {
  std::ifstream file(path);
  if (!file) {
    error = std::string("cannot open ") + path;
    return false;
  }
  file.seekg(0, std::ios::end);
  std::streamsize const size = file.tellg();
  if (size < 0) {
    error = std::string("cannot read ") + path;
    return false;
  }
  if (static_cast<std::uint64_t>(size) > kMaxAssetFileSize) {
    error = std::string("model too large (over 2 GiB): ") + path;
    return false;
  }
  file.seekg(0);
  std::ostringstream content;
  content << file.rdbuf();
  std::string text = content.str();
  if (text.empty()) {
    error = std::string("empty or unreadable ") + path;
    return false;
  }

  // Look for `mtllib` and load the sibling .mtl for usemtl materials.
  MtlMaterials materials;
  std::string directory(path);
  std::size_t slash = directory.find_last_of('/');
  directory = slash == std::string::npos ? std::string{} : directory.substr(0, slash + 1U);
  std::istringstream scan(text);
  std::string line;
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

  // Resolve map_Kd / map_Bump images (missing or undecodable images load
  // untextured / flat). The same path shared by several materials is decoded
  // once.
  std::vector<ModelTexture> textures;
  std::vector<std::pair<std::string, std::int32_t>> resolvedImages;
  auto resolveImage = [&](std::string const& path, char const* what) -> std::int32_t {
    for (std::pair<std::string, std::int32_t> const& entry : resolvedImages) {
      if (entry.first == path) {
        return entry.second;
      }
    }
    std::ifstream imageFile(directory + path, std::ios::binary);
    if (!withImages) {
      // Geometry-only preview load: keep the texture entry (mesh indices stay
      // stable), skip all image I/O and decoding.
      ModelTexture texture;
      textures.push_back(std::move(texture));
      std::int32_t const handle = static_cast<std::int32_t>(textures.size() - 1U);
      resolvedImages.push_back({path, handle});
      return handle;
    }
    if (!imageFile) {
      std::printf("obj: %s %s not found; skipped\n", what, path.c_str());
      return -1;
    }
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(imageFile)),
                                    std::istreambuf_iterator<char>());
    ModelTexture texture;
    if (!decodeImage(bytes.data(), bytes.size(), texture.image, error)) {
      std::printf("obj: %s %s failed to decode (%s); skipped\n", what, path.c_str(),
                  error.c_str());
      error.clear();
      return -1;
    }
    textures.push_back(std::move(texture));
    std::int32_t handle = static_cast<std::int32_t>(textures.size() - 1U);
    resolvedImages.push_back({path, handle});
    return handle;
  };
  for (MtlMaterial& material : materials) {
    if (!material.mapKd.empty()) {
      material.texture = resolveImage(material.mapKd, "texture");
    }
    if (!material.mapBump.empty()) {
      material.normalTexture = resolveImage(material.mapBump, "normal map");
    }
  }

  return loadObj(text, materials, textures, out, error);
}

} // namespace vredit
