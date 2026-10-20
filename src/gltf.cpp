#include "gltf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

#include "image.h"
#include "json.h"

namespace vredit {

namespace {

constexpr std::uint32_t kGlbMagic = 0x46546C67U; // 'glTF'
constexpr std::uint32_t kChunkJson = 0x4E4F534AU;
constexpr std::uint32_t kChunkBin = 0x004E4942U;
constexpr std::uint32_t kMaxNodeDepth = 1024U;

std::uint32_t readU32(std::uint8_t const* p) {
  std::uint32_t value = 0;
  std::memcpy(&value, p, 4);
  return value;
}

struct BufferView {
  std::uint8_t const* data{nullptr};
  std::size_t length{0};
  std::size_t stride{0};
};

struct AccessorView {
  std::uint8_t const* data{nullptr};
  std::size_t byteStride{0};
  std::size_t elementSize{0};
  std::size_t count{0};
  std::uint32_t componentType{0};
  std::uint32_t components{0};
  bool normalised{false};
};

std::size_t componentSize(std::uint32_t componentType) {
  switch (componentType) {
  case 5120: // BYTE
  case 5121: // UNSIGNED_BYTE
    return 1;
  case 5122: // SHORT
  case 5123: // UNSIGNED_SHORT
    return 2;
  case 5125: // UNSIGNED_INT
  case 5126: // FLOAT
    return 4;
  default:
    return 0;
  }
}

bool typeComponents(std::string const& type, std::uint32_t& components) {
  if (type == "SCALAR") {
    components = 1;
  } else if (type == "VEC2") {
    components = 2;
  } else if (type == "VEC3") {
    components = 3;
  } else if (type == "VEC4") {
    components = 4;
  } else {
    return false;
  }
  return true;
}

float readComponentFloat(AccessorView const& view, std::size_t element, std::uint32_t component) {
  std::uint8_t const* p =
      view.data + element * view.byteStride + component * componentSize(view.componentType);
  switch (view.componentType) {
  case 5126: {
    float value = 0.0F;
    std::memcpy(&value, p, 4);
    return value;
  }
  case 5125: {
    std::uint32_t value = 0;
    std::memcpy(&value, p, 4);
    return view.normalised ? static_cast<float>(value) / 4294967295.0F : static_cast<float>(value);
  }
  case 5123: {
    std::uint16_t value = 0;
    std::memcpy(&value, p, 2);
    return view.normalised ? static_cast<float>(value) / 65535.0F : static_cast<float>(value);
  }
  case 5121:
    return view.normalised ? static_cast<float>(*p) / 255.0F : static_cast<float>(*p);
  case 5122: {
    std::int16_t value = 0;
    std::memcpy(&value, p, 2);
    return view.normalised ? std::fmax(static_cast<float>(value) / 32767.0F, -1.0F)
                           : static_cast<float>(value);
  }
  case 5120: {
    std::int8_t value = 0;
    std::memcpy(&value, p, 1);
    return view.normalised ? std::fmax(static_cast<float>(value) / 127.0F, -1.0F)
                           : static_cast<float>(value);
  }
  default:
    return 0.0F;
  }
}

std::uint32_t readIndexScalar(AccessorView const& view, std::size_t element) {
  std::uint8_t const* p = view.data + element * view.byteStride;
  switch (view.componentType) {
  case 5125: {
    std::uint32_t value = 0;
    std::memcpy(&value, p, 4);
    return value;
  }
  case 5123: {
    std::uint16_t value = 0;
    std::memcpy(&value, p, 2);
    return value;
  }
  case 5121:
    return *p;
  default:
    return 0;
  }
}

class GlbReader {
public:
  GlbReader(std::vector<std::uint8_t> const& bytes,
            Model& model,
            std::string& error,
            bool withImages)
      : bytes_(bytes), model_(model), error_(error), withImages_(withImages) {}

  bool run();

private:
  bool fail(char const* message);
  // Reads a JSON number as a non-negative, integral size. Rejects negatives,
  // fractions and values that do not fit, so index/offset maths cannot wrap.
  bool readSize(JsonDocument::NodeId node, std::size_t& out, char const* what);
  bool readOptionalSize(JsonDocument::NodeId node, std::size_t& out, char const* what);
  bool parseContainer();
  bool buildBufferViews(JsonDocument::NodeId bufferViewsNode);
  bool buildAccessor(JsonDocument::NodeId accessorNode, AccessorView& out);
  bool addPrimitive(JsonDocument::NodeId primitiveNode, std::int32_t& meshIndexOut);
  bool collectNode(std::size_t nodeIndex, std::int32_t parentModel, std::int32_t& modelIndexOut,
                   std::uint32_t depth);
  // Fills material colour, texture index and alpha on the mesh. Bad material/
  // texture references and undecodable images degrade to untextured with a
  // warning (geometry is still useful); only geometry-maths errors fail.
  bool buildMaterial(JsonDocument::NodeId primitiveNode, ModelMesh& mesh);
  // Binds glTF texture `textureIndex` (image + sampler wrap modes) into a mesh
  // slot, decoding and deduping on (image, wrap modes). Bad references leave
  // the slot unbound.
  void bindTexture(std::size_t textureIndex, std::int32_t& slotOut);
  Transform parseTransform(JsonDocument::NodeId nodeDef);

  std::vector<std::uint8_t> const& bytes_;
  Model& model_;
  std::string& error_;
  bool withImages_{true}; // false = geometry-only preview load (no image decode)
  JsonDocument document_;
  std::uint8_t const* bin_{nullptr};
  std::size_t binLength_{0};
  std::vector<BufferView> bufferViews_;
  JsonDocument::NodeId accessorsNode_{JsonDocument::kInvalid};
  JsonDocument::NodeId meshesNode_{JsonDocument::kInvalid};
  JsonDocument::NodeId materialsNode_{JsonDocument::kInvalid};
  JsonDocument::NodeId nodesNode_{JsonDocument::kInvalid};
  JsonDocument::NodeId texturesNode_{JsonDocument::kInvalid};
  JsonDocument::NodeId samplersNode_{JsonDocument::kInvalid};
  JsonDocument::NodeId imagesNode_{JsonDocument::kInvalid};
  struct TextureKey {
    std::size_t source{0};
    bool repeatU{true};
    bool repeatV{true};
    std::int32_t texture{-1};
  };
  std::vector<TextureKey> textureCache_; // dedupe on (image, wrap modes)
  std::vector<bool> visited_;
};

bool GlbReader::fail(char const* message) {
  error_ = message;
  return false;
}

bool GlbReader::readSize(JsonDocument::NodeId node, std::size_t& out, char const* what) {
  if (document_.type(node) != JsonDocument::Type::Number) {
    error_ = std::string("glb: missing or non-number ") + what;
    return false;
  }
  double value = document_.asNumber(node);
  if (!(value >= 0.0) || std::floor(value) != value || value >= 18446744073709551616.0) {
    error_ = std::string("glb: bad ") + what;
    return false;
  }
  out = static_cast<std::size_t>(value);
  return true;
}

bool GlbReader::readOptionalSize(JsonDocument::NodeId node, std::size_t& out, char const* what) {
  if (document_.type(node) != JsonDocument::Type::Number) {
    return true; // absent: caller's default stands
  }
  return readSize(node, out, what);
}

bool GlbReader::parseContainer() {
  if (bytes_.size() < 12U) {
    return fail("glb: file too small");
  }
  std::uint8_t const* header = bytes_.data();
  if (readU32(header) != kGlbMagic) {
    return fail("glb: bad magic");
  }
  if (readU32(header + 4) != 2U) {
    return fail("glb: unsupported version");
  }
  std::uint32_t declaredLength = readU32(header + 8);
  if (declaredLength > bytes_.size()) {
    return fail("glb: truncated file");
  }

  std::size_t offset = 12U;
  bool sawJson = false;
  while (offset + 8U <= declaredLength) {
    std::uint32_t chunkLength = readU32(bytes_.data() + offset);
    std::uint32_t chunkType = readU32(bytes_.data() + offset + 4U);
    offset += 8U;
    if (offset + chunkLength > declaredLength) {
      return fail("glb: truncated chunk");
    }
    if (chunkType == kChunkJson && !sawJson) {
      std::string jsonError;
      std::string_view jsonText(reinterpret_cast<char const*>(bytes_.data() + offset), chunkLength);
      if (!document_.parse(jsonText, jsonError)) {
        error_ = jsonError;
        return false;
      }
      sawJson = true;
    } else if (chunkType == kChunkBin && bin_ == nullptr) {
      bin_ = bytes_.data() + offset;
      binLength_ = chunkLength;
    }
    offset += chunkLength;
  }
  if (!sawJson) {
    return fail("glb: missing JSON chunk");
  }
  return true;
}

bool GlbReader::buildBufferViews(JsonDocument::NodeId bufferViewsNode) {
  std::size_t count = document_.size(bufferViewsNode);
  bufferViews_.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    JsonDocument::NodeId viewNode = document_.element(bufferViewsNode, i);
    std::size_t bufferIndex = 0;
    if (!readSize(document_.member(viewNode, "buffer"), bufferIndex, "buffer index")) {
      return false;
    }
    if (bufferIndex != 0U || bin_ == nullptr) {
      return fail("glb: only the embedded BIN buffer is supported");
    }
    std::size_t byteOffset = 0;
    std::size_t byteLength = 0;
    if (!readOptionalSize(document_.member(viewNode, "byteOffset"), byteOffset, "byteOffset") ||
        !readSize(document_.member(viewNode, "byteLength"), byteLength, "byteLength")) {
      return false;
    }
    // Overflow-safe range check against the BIN chunk.
    if (byteOffset > binLength_ || byteLength > binLength_ - byteOffset) {
      return fail("glb: bufferView out of range");
    }
    BufferView& view = bufferViews_[i];
    view.data = bin_ + byteOffset;
    view.length = byteLength;
    if (!readOptionalSize(document_.member(viewNode, "byteStride"), view.stride, "byteStride")) {
      return false;
    }
  }
  return true;
}

bool GlbReader::buildAccessor(JsonDocument::NodeId accessorNode, AccessorView& out) {
  if (accessorNode == JsonDocument::kInvalid ||
      document_.type(accessorNode) != JsonDocument::Type::Object) {
    return fail("glb: bad accessor reference");
  }
  std::size_t componentType = 0;
  std::size_t count = 0;
  if (!readSize(document_.member(accessorNode, "componentType"), componentType, "componentType") ||
      !readSize(document_.member(accessorNode, "count"), count, "count")) {
    return false;
  }
  std::uint32_t components = 0;
  if (!typeComponents(document_.asString(document_.member(accessorNode, "type")), components)) {
    return fail("glb: unsupported accessor type");
  }
  std::size_t compSize = componentSize(static_cast<std::uint32_t>(componentType));
  if (compSize == 0U) {
    return fail("glb: unsupported componentType");
  }
  std::size_t elementSize = compSize * components;

  JsonDocument::NodeId viewIndexNode = document_.member(accessorNode, "bufferView");
  if (document_.type(viewIndexNode) != JsonDocument::Type::Number) {
    return fail("glb: accessor without bufferView is unsupported");
  }
  std::size_t viewIndex = 0;
  if (!readSize(viewIndexNode, viewIndex, "bufferView index")) {
    return false;
  }
  if (viewIndex >= bufferViews_.size()) {
    return fail("glb: accessor bufferView out of range");
  }
  BufferView const& view = bufferViews_[viewIndex];
  std::size_t byteOffset = 0;
  if (!readOptionalSize(document_.member(accessorNode, "byteOffset"), byteOffset, "byteOffset")) {
    return false;
  }

  std::size_t stride = view.stride != 0U ? view.stride : elementSize;
  if (stride < elementSize) {
    return fail("glb: byteStride smaller than element size");
  }
  // Overflow-safe span check: first element must fit, then the last element's
  // start offset must fit in the remaining space.
  if (byteOffset > view.length ||
      (count > 0U && view.length - byteOffset < elementSize)) {
    return fail("glb: accessor out of range");
  }
  if (count > 1U && (count - 1U) > (view.length - byteOffset - elementSize) / stride) {
    return fail("glb: accessor out of range");
  }

  out.data = view.data + byteOffset;
  out.byteStride = stride;
  out.elementSize = elementSize;
  out.count = count;
  out.componentType = static_cast<std::uint32_t>(componentType);
  out.components = components;
  out.normalised = document_.asBool(document_.member(accessorNode, "normalized"));
  return true;
}

Transform GlbReader::parseTransform(JsonDocument::NodeId nodeDef) {
  Transform transform;
  JsonDocument::NodeId matrixNode = document_.member(nodeDef, "matrix");
  if (document_.type(matrixNode) == JsonDocument::Type::Array &&
      document_.size(matrixNode) == 16U) {
    Mat4 matrix;
    for (std::size_t i = 0; i < 16U; ++i) {
      matrix.m[i] = static_cast<float>(document_.asNumber(document_.element(matrixNode, i)));
    }
    decompose(matrix, transform.translation, transform.rotation, transform.scale);
    return transform;
  }

  JsonDocument::NodeId translation = document_.member(nodeDef, "translation");
  if (document_.type(translation) == JsonDocument::Type::Array) {
    transform.translation = {
        static_cast<float>(document_.asNumber(document_.element(translation, 0))),
        static_cast<float>(document_.asNumber(document_.element(translation, 1))),
        static_cast<float>(document_.asNumber(document_.element(translation, 2)))};
  }
  JsonDocument::NodeId rotation = document_.member(nodeDef, "rotation");
  if (document_.type(rotation) == JsonDocument::Type::Array) {
    // glTF quaternions are [x, y, z, w] — same order as Quat.
    transform.rotation = {
        static_cast<float>(document_.asNumber(document_.element(rotation, 0))),
        static_cast<float>(document_.asNumber(document_.element(rotation, 1))),
        static_cast<float>(document_.asNumber(document_.element(rotation, 2))),
        static_cast<float>(document_.asNumber(document_.element(rotation, 3)))};
  }
  JsonDocument::NodeId scale = document_.member(nodeDef, "scale");
  if (document_.type(scale) == JsonDocument::Type::Array) {
    transform.scale = {
        static_cast<float>(document_.asNumber(document_.element(scale, 0))),
        static_cast<float>(document_.asNumber(document_.element(scale, 1))),
        static_cast<float>(document_.asNumber(document_.element(scale, 2)))};
  }
  return transform;
}

bool GlbReader::buildMaterial(JsonDocument::NodeId primitiveNode, ModelMesh& mesh) {
  // Colour and alpha are cosmetic: bad references fall back to defaults.
  JsonDocument::NodeId materialIndexNode = document_.member(primitiveNode, "material");
  if (document_.type(materialIndexNode) != JsonDocument::Type::Number) {
    return true;
  }
  std::size_t materialIndex = 0;
  if (!readOptionalSize(materialIndexNode, materialIndex, "material")) {
    error_.clear();
    return true;
  }
  JsonDocument::NodeId materialNode = document_.element(materialsNode_, materialIndex);
  JsonDocument::NodeId pbr = document_.member(materialNode, "pbrMetallicRoughness");
  JsonDocument::NodeId factor = document_.member(pbr, "baseColorFactor");
  if (document_.type(factor) == JsonDocument::Type::Array && document_.size(factor) >= 3U) {
    mesh.colour = {
        static_cast<float>(document_.asNumber(document_.element(factor, 0))),
        static_cast<float>(document_.asNumber(document_.element(factor, 1))),
        static_cast<float>(document_.asNumber(document_.element(factor, 2)))};
  }

  JsonDocument::NodeId alphaModeNode = document_.member(materialNode, "alphaMode");
  std::string const& alphaMode = document_.asString(alphaModeNode);
  if (alphaMode == "MASK") {
    mesh.alpha = AlphaMode::Mask;
  } else if (alphaMode == "BLEND") {
    mesh.alpha = AlphaMode::Blend;
  }
  JsonDocument::NodeId cutoffNode = document_.member(materialNode, "alphaCutoff");
  if (document_.type(cutoffNode) == JsonDocument::Type::Number) {
    mesh.alphaCutoff = static_cast<float>(document_.asNumber(cutoffNode));
  }
  if (document_.type(factor) == JsonDocument::Type::Array && document_.size(factor) >= 4U) {
    mesh.alphaFactor = static_cast<float>(document_.asNumber(document_.element(factor, 3)));
  }

  JsonDocument::NodeId baseColorTexture = document_.member(pbr, "baseColorTexture");
  if (document_.type(document_.member(baseColorTexture, "index")) == JsonDocument::Type::Number) {
    std::size_t textureIndex = 0;
    if (readOptionalSize(document_.member(baseColorTexture, "index"), textureIndex,
                         "baseColorTexture index")) {
      bindTexture(textureIndex, mesh.texture);
    } else {
      error_.clear();
    }
  }

  // Tangent-space normal map (core glTF material.normalTexture; `scale` tilts
  // the xy strength).
  JsonDocument::NodeId normalTexture = document_.member(materialNode, "normalTexture");
  if (document_.type(document_.member(normalTexture, "index")) == JsonDocument::Type::Number) {
    std::size_t textureIndex = 0;
    if (readOptionalSize(document_.member(normalTexture, "index"), textureIndex,
                         "normalTexture index")) {
      bindTexture(textureIndex, mesh.normalTexture);
    } else {
      error_.clear();
    }
    JsonDocument::NodeId scaleNode = document_.member(normalTexture, "scale");
    if (document_.type(scaleNode) == JsonDocument::Type::Number) {
      mesh.normalScale = static_cast<float>(document_.asNumber(scaleNode));
    }
  }
  return true;
}

void GlbReader::bindTexture(std::size_t textureIndex, std::int32_t& slotOut) {
  JsonDocument::NodeId textureNode = document_.element(texturesNode_, textureIndex);
  if (textureNode == JsonDocument::kInvalid) {
    return;
  }
  std::size_t sourceIndex = 0;
  if (!readSize(document_.member(textureNode, "source"), sourceIndex, "texture source")) {
    error_.clear();
    return;
  }
  if (sourceIndex >= document_.size(imagesNode_)) {
    return;
  }

  // Sampler wrap modes (mirrored-repeat approximates to repeat). A bad sampler
  // reference is cosmetic: default to repeat.
  bool repeatU = true;
  bool repeatV = true;
  std::size_t samplerIndex = 0;
  if (!readOptionalSize(document_.member(textureNode, "sampler"), samplerIndex, "sampler")) {
    error_.clear();
  } else if (document_.type(document_.member(textureNode, "sampler")) ==
             JsonDocument::Type::Number) {
    JsonDocument::NodeId samplerNode = document_.element(samplersNode_, samplerIndex);
    JsonDocument::NodeId wrapS = document_.member(samplerNode, "wrapS");
    JsonDocument::NodeId wrapT = document_.member(samplerNode, "wrapT");
    if (document_.type(wrapS) == JsonDocument::Type::Number) {
      repeatU = document_.asNumber(wrapS) != 33071.0;
    }
    if (document_.type(wrapT) == JsonDocument::Type::Number) {
      repeatV = document_.asNumber(wrapT) != 33071.0;
    }
  }

  // glTF texture = image + sampler: same image with different wrap modes is a
  // different texture, so the cache keys on both.
  for (TextureKey const& key : textureCache_) {
    if (key.source == sourceIndex && key.repeatU == repeatU && key.repeatV == repeatV) {
      slotOut = key.texture;
      return;
    }
  }

  JsonDocument::NodeId imageNode = document_.element(imagesNode_, sourceIndex);
  JsonDocument::NodeId viewIndexNode = document_.member(imageNode, "bufferView");
  std::size_t viewIndex = 0;
  if (document_.type(viewIndexNode) != JsonDocument::Type::Number ||
      !readSize(viewIndexNode, viewIndex, "image bufferView") || viewIndex >= bufferViews_.size()) {
    error_.clear();
    std::printf("glb: image %zu not embedded; texture skipped\n", sourceIndex);
    return;
  }
  BufferView const& view = bufferViews_[viewIndex];
  ModelTexture texture;
  texture.repeatU = repeatU;
  texture.repeatV = repeatV;
  if (!withImages_) {
    // Geometry-only preview load: keep the texture ENTRY so mesh texture
    // indices stay stable, but skip the image entirely (the empty image
    // registers as untextured). Previews cost megabytes, not gigabytes.
    model_.textures.push_back(std::move(texture));
    std::int32_t handle = static_cast<std::int32_t>(model_.textures.size() - 1U);
    textureCache_.push_back({sourceIndex, repeatU, repeatV, handle});
    slotOut = handle;
    return;
  }
  if (!decodeImage(view.data, view.length, texture.image, error_)) {
    std::printf("glb: image %zu failed to decode (%s); texture skipped\n", sourceIndex,
                error_.c_str());
    error_.clear();
    return;
  }
  model_.textures.push_back(std::move(texture));
  std::int32_t handle = static_cast<std::int32_t>(model_.textures.size() - 1U);
  textureCache_.push_back({sourceIndex, repeatU, repeatV, handle});
  slotOut = handle;
}

bool GlbReader::addPrimitive(JsonDocument::NodeId primitiveNode, std::int32_t& meshIndexOut) {
  JsonDocument::NodeId modeNode = document_.member(primitiveNode, "mode");
  if (document_.type(modeNode) == JsonDocument::Type::Number &&
      document_.asNumber(modeNode) != 4.0) {
    return fail("glb: only TRIANGLES primitives are supported");
  }

  JsonDocument::NodeId attributes = document_.member(primitiveNode, "attributes");
  std::size_t positionIndex = 0;
  if (!readSize(document_.member(attributes, "POSITION"), positionIndex, "POSITION accessor")) {
    return false;
  }

  AccessorView positions{};
  if (!buildAccessor(document_.element(accessorsNode_, positionIndex), positions)) {
    return false;
  }
  if (positions.components != 3U) {
    return fail("glb: POSITION must be VEC3");
  }

  AccessorView normals{};
  bool hasNormals = false;
  std::size_t normalIndex = 0;
  if (readOptionalSize(document_.member(attributes, "NORMAL"), normalIndex, "NORMAL accessor") &&
      document_.type(document_.member(attributes, "NORMAL")) == JsonDocument::Type::Number) {
    if (!buildAccessor(document_.element(accessorsNode_, normalIndex), normals)) {
      return false;
    }
    hasNormals = normals.components == 3U && normals.count == positions.count;
  }

  AccessorView colours{};
  bool hasColours = false;
  std::size_t colourIndex = 0;
  if (readOptionalSize(document_.member(attributes, "COLOR_0"), colourIndex, "COLOR_0 accessor") &&
      document_.type(document_.member(attributes, "COLOR_0")) == JsonDocument::Type::Number) {
    if (!buildAccessor(document_.element(accessorsNode_, colourIndex), colours)) {
      return false;
    }
    hasColours = colours.components >= 3U && colours.count == positions.count;
  }

  AccessorView texcoords{};
  bool hasUvs = false;
  std::size_t texcoordIndex = 0;
  if (readOptionalSize(document_.member(attributes, "TEXCOORD_0"), texcoordIndex,
                       "TEXCOORD_0 accessor") &&
      document_.type(document_.member(attributes, "TEXCOORD_0")) == JsonDocument::Type::Number) {
    if (!buildAccessor(document_.element(accessorsNode_, texcoordIndex), texcoords)) {
      return false;
    }
    hasUvs = texcoords.components >= 2U && texcoords.count == positions.count;
  }

  ModelMesh mesh;
  if (!buildMaterial(primitiveNode, mesh)) {
    return false;
  }
  mesh.data.vertices.resize(positions.count);
  for (std::size_t v = 0; v < positions.count; ++v) {
    MeshVertex& vertex = mesh.data.vertices[v];
    vertex.pos = {readComponentFloat(positions, v, 0), readComponentFloat(positions, v, 1),
                  readComponentFloat(positions, v, 2)};
    vertex.colour = hasColours
                        ? Vec3{readComponentFloat(colours, v, 0), readComponentFloat(colours, v, 1),
                               readComponentFloat(colours, v, 2)}
                        : Vec3{1.0F, 1.0F, 1.0F};
    vertex.normal = hasNormals
                        ? Vec3{readComponentFloat(normals, v, 0), readComponentFloat(normals, v, 1),
                               readComponentFloat(normals, v, 2)}
                        : Vec3{0.0F, 0.0F, 0.0F};
    vertex.uv = hasUvs ? Vec2{readComponentFloat(texcoords, v, 0),
                              readComponentFloat(texcoords, v, 1)}
                       : Vec2{0.0F, 0.0F};
  }

  std::size_t indicesIndex = 0;
  if (readOptionalSize(document_.member(primitiveNode, "indices"), indicesIndex, "indices") &&
      document_.type(document_.member(primitiveNode, "indices")) == JsonDocument::Type::Number) {
    AccessorView indices{};
    if (!buildAccessor(document_.element(accessorsNode_, indicesIndex), indices)) {
      return false;
    }
    if (indices.components != 1U) {
      return fail("glb: indices accessor must be SCALAR");
    }
    mesh.data.indices.resize(indices.count);
    for (std::size_t i = 0; i < indices.count; ++i) {
      std::uint32_t index = readIndexScalar(indices, i);
      if (index >= positions.count) {
        return fail("glb: index out of range");
      }
      mesh.data.indices[i] = index;
    }
  } else {
    mesh.data.indices.resize(positions.count);
    for (std::size_t i = 0; i < positions.count; ++i) {
      mesh.data.indices[i] = static_cast<std::uint32_t>(i);
    }
  }

  // Generate smoothed normals when the file has none.
  if (!hasNormals) {
    for (std::size_t i = 0; i + 2U < mesh.data.indices.size(); i += 3U) {
      MeshVertex& a = mesh.data.vertices[mesh.data.indices[i]];
      MeshVertex& b = mesh.data.vertices[mesh.data.indices[i + 1U]];
      MeshVertex& c = mesh.data.vertices[mesh.data.indices[i + 2U]];
      Vec3 normal = cross(b.pos - a.pos, c.pos - a.pos);
      a.normal = a.normal + normal;
      b.normal = b.normal + normal;
      c.normal = c.normal + normal;
    }
    for (MeshVertex& vertex : mesh.data.vertices) {
      vertex.normal = length(vertex.normal) > 1e-8F ? normalized(vertex.normal)
                                                    : Vec3{0.0F, 1.0F, 0.0F};
    }
  }

  model_.meshes.push_back(std::move(mesh));
  meshIndexOut = static_cast<std::int32_t>(model_.meshes.size() - 1U);
  return true;
}

bool GlbReader::collectNode(std::size_t nodeIndex, std::int32_t parentModel,
                            std::int32_t& modelIndexOut, std::uint32_t depth) {
  if (depth > kMaxNodeDepth) {
    return fail("glb: node hierarchy too deep");
  }
  std::size_t nodeCount = document_.size(nodesNode_);
  if (nodeIndex >= nodeCount) {
    return fail("glb: node index out of range");
  }
  if (visited_[nodeIndex]) {
    return fail("glb: node referenced more than once (not a tree)");
  }
  visited_[nodeIndex] = true;

  JsonDocument::NodeId nodeDef = document_.element(nodesNode_, nodeIndex);
  ModelNode node;
  node.name = document_.asString(document_.member(nodeDef, "name"));
  node.local = parseTransform(nodeDef);
  node.parent = parentModel;
  model_.nodes.push_back(node);
  std::int32_t modelIndex = static_cast<std::int32_t>(model_.nodes.size() - 1U);
  modelIndexOut = modelIndex;
  if (parentModel < 0) {
    model_.roots.push_back(modelIndex);
  }

  JsonDocument::NodeId meshIndexNode = document_.member(nodeDef, "mesh");
  if (document_.type(meshIndexNode) == JsonDocument::Type::Number) {
    std::size_t meshIndex = 0;
    if (!readSize(meshIndexNode, meshIndex, "mesh index")) {
      return false;
    }
    JsonDocument::NodeId meshDef = document_.element(meshesNode_, meshIndex);
    JsonDocument::NodeId primitives = document_.member(meshDef, "primitives");
    std::size_t primitiveCount = document_.size(primitives);
    std::vector<std::int32_t> primitiveMeshes;
    for (std::size_t p = 0; p < primitiveCount; ++p) {
      std::int32_t meshOut = -1;
      if (!addPrimitive(document_.element(primitives, p), meshOut)) {
        return false;
      }
      primitiveMeshes.push_back(meshOut);
    }
    if (primitiveMeshes.size() == 1U) {
      model_.nodes[static_cast<std::size_t>(modelIndex)].mesh = primitiveMeshes[0];
    } else {
      // Multi-primitive mesh: wrapper node keeps the transform, one child per
      // primitive shares it.
      for (std::size_t p = 0; p < primitiveMeshes.size(); ++p) {
        ModelNode child;
        child.name = node.name + "/primitive" + std::to_string(p);
        child.parent = modelIndex;
        child.mesh = primitiveMeshes[p];
        model_.nodes.push_back(std::move(child));
      }
    }
  }

  JsonDocument::NodeId children = document_.member(nodeDef, "children");
  for (std::size_t c = 0; c < document_.size(children); ++c) {
    std::size_t childIndex = 0;
    if (!readSize(document_.element(children, c), childIndex, "child node index")) {
      return false;
    }
    std::int32_t childModel = -1;
    if (!collectNode(childIndex, modelIndex, childModel, depth + 1U)) {
      return false;
    }
  }
  return true;
}

bool GlbReader::run() {
  if (!parseContainer()) {
    return false;
  }
  JsonDocument::NodeId root = document_.root();
  if (!buildBufferViews(document_.member(root, "bufferViews"))) {
    return false;
  }
  accessorsNode_ = document_.member(root, "accessors");
  meshesNode_ = document_.member(root, "meshes");
  materialsNode_ = document_.member(root, "materials");
  nodesNode_ = document_.member(root, "nodes");
  texturesNode_ = document_.member(root, "textures");
  samplersNode_ = document_.member(root, "samplers");
  imagesNode_ = document_.member(root, "images");
  visited_.assign(document_.size(nodesNode_), false);

  JsonDocument::NodeId scenesNode = document_.member(root, "scenes");
  JsonDocument::NodeId sceneNode = JsonDocument::kInvalid;
  std::size_t sceneIndex = 0;
  if (readOptionalSize(document_.member(root, "scene"), sceneIndex, "scene index")) {
    sceneNode = document_.element(scenesNode, sceneIndex);
  }
  if (sceneNode == JsonDocument::kInvalid && document_.size(scenesNode) > 0U) {
    sceneNode = document_.element(scenesNode, 0U);
  }
  if (sceneNode == JsonDocument::kInvalid) {
    return fail("glb: no scene");
  }

  JsonDocument::NodeId sceneRoots = document_.member(sceneNode, "nodes");
  for (std::size_t r = 0; r < document_.size(sceneRoots); ++r) {
    std::size_t nodeIndex = 0;
    if (!readSize(document_.element(sceneRoots, r), nodeIndex, "root node index")) {
      return false;
    }
    std::int32_t modelIndex = -1;
    if (!collectNode(nodeIndex, -1, modelIndex, 0U)) {
      return false;
    }
  }
  return true;
}

} // namespace

bool loadGlb(std::vector<std::uint8_t> const& bytes,
             Model& out,
             std::string& error,
             bool withImages) {
  out = Model{};
  GlbReader reader(bytes, out, error, withImages);
  if (!reader.run()) {
    out = Model{}; // defined state on failure
    return false;
  }
  return true;
}

// The size cap (kMaxAssetFileSize, image.h) guards hostile files — see there.
bool loadGlbFile(char const* path, Model& out, std::string& error, bool withImages) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    error = std::string("cannot open ") + path;
    return false;
  }
  std::streamsize size = file.tellg();
  if (size < 0) {
    error = std::string("cannot read ") + path;
    return false;
  }
  if (static_cast<std::uint64_t>(size) > kMaxAssetFileSize) {
    error = std::string("model too large (over 2 GiB): ") + path;
    return false;
  }
  file.seekg(0);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  if (size > 0 && !file.read(reinterpret_cast<char*>(bytes.data()), size)) {
    error = std::string("cannot read ") + path;
    return false;
  }
  return loadGlb(bytes, out, error, withImages);
}

} // namespace vredit
