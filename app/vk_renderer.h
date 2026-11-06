#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "camera.h"
#include "geometry.h"
#include "lighting.h"
#include "mesh.h"

namespace vredit {

// Vulkan viewport renderer: draws registered meshes per DrawItem, depth-tested
// lines (floor grid) and always-on-top overlay lines (gizmo handles). Vulkan
// 1.3 with dynamic rendering. Vulkan types appear in the external-device (XR)
// API only.
class VkRenderer {
public:
  struct Config {
    bool windowed{false};
    void* xcbConnection{nullptr}; // xcb_connection_t*, windowed only
    std::uint32_t xcbWindow{0};   // windowed only
    std::uint32_t width{1280};
    std::uint32_t height{800};
    char const* shaderDir{".build/shaders"};
  };

  VkRenderer();
  ~VkRenderer();
  VkRenderer(VkRenderer const&) = delete;
  VkRenderer& operator=(VkRenderer const&) = delete;

  bool init(Config const& config);
  // Device loss (or unrecoverable renderer synchronisation) requires shutdown.
  // If rollback fails, retain its token and retry; do not continue as if it
  // succeeded. A fatal error requires stopping rendering and shutting down.
  bool hasFatalError() const;
  void shutdown();

  bool resize(std::uint32_t width, std::uint32_t height);

  // Registers mesh data and returns a handle (0 is the built-in cube).
  // `dynamic` allocates a persistent host staging slot for updateMeshVertices;
  // static meshes keep the cheaper upload-once path. `trackTriangleSums`
  // keeps the per-triangle depth keys used to sort alpha-blended draws; pass
  // false for material that is always opaque, which makes each dynamic update
  // skip an O(triangles) pass (a blended draw of such a mesh then uses index
  // order, as every unsortable mesh already does). Safe to call at any time;
  // internally synchronises with in-flight frames. C++ allocation exceptions
  // propagate; a failed registration releases its unregistered resources.
  MeshHandle addMesh(MeshData const& data, bool dynamic = false,
                     bool trackTriangleSums = true);

  // Replaces the position and normal data of a dynamic mesh without changing
  // its topology or vertex count. The upload is queued for the next draw,
  // which already waits for the previous frame's fence: no vkDeviceWaitIdle
  // and no per-frame GPU stall. Blend sorting keys are recomputed from the new
  // vertices when the mesh tracks them. `boundsCentre`, when non-null, is the
  // new mesh-space bounds centre used to order blended draws; passing the
  // value the caller already computed saves a full read of the vertices (it
  // only affects blend ordering, never geometry). Returns false (and leaves
  // the old GPU data intact) for a static mesh, a vertex-count change,
  // non-finite data or a Vulkan failure. Allocation exceptions propagate and
  // leave the previous pending pose and blend ordering unchanged.
  bool updateMeshVertices(MeshHandle handle, std::vector<MeshVertex> const& vertices,
                          Vec3 const* boundsCentre = nullptr);

  // Two reusable opaque hand meshes, depth-tested against scene geometry
  // and each other. slot is 0 (left) or 1
  // (right); each has a stable handle and fixed capacity. The caller adds a
  // DrawItem with that handle to each eye/frame; its material is forced opaque.
  // Hands share opaque/masked scene depth; translucent scene geometry is
  // blended afterwards. Overlay lines draw on top. Hands are clip-volume
  // limited. Empty
  // vertex and index arrays clear the slot. Invalid input or a
  // Vulkan failure returns kInvalidMesh and leaves the previous data intact.
  // Update is fence-synchronised; call on the renderer thread, before drawing.
  static constexpr std::size_t kHandMeshSlots = 2U;
  static constexpr std::size_t kHandMeshMaxVertices = 65536U;
  static constexpr std::size_t kHandMeshMaxIndices = 196608U;
  MeshHandle updateHandMesh(std::size_t slot, MeshData const& data);

  // Registers an RGBA8 texture and returns a handle (0 is the built-in white
  // fallback). Up to 256 textures; safe to call at any time. C++ allocation
  // exceptions propagate; a failed registration releases its resources.
  TextureHandle addTexture(Image const& image, bool repeatU, bool repeatV);

  // One active checkpoint per renderer, on the renderer thread. A default or
  // rejected token is invalid. Copies refer to the same transaction; commit or
  // rollback succeeds once only and rejects tokens from other renderers.
  // Shutdown invalidates tokens. Rollback waits for the device, restores removed
  // texture descriptor slots to white and frees meshes, textures and hand slots
  // first registered since the checkpoint. In-place updates to existing assets
  // and other renderer state are not reverted. Rolled-back handles can be reused:
  // callers must discard them. A failed GPU wait leaves the transaction active; retry unless hasFatalError().
  class ResourceCheckpoint {
  public:
    ResourceCheckpoint() = default;
    // Reports whether creation succeeded, not whether a copy was consumed.
    explicit operator bool() const { return static_cast<bool>(identity_); }
  private:
    friend class VkRenderer;
    std::shared_ptr<void> identity_;
  };
  ResourceCheckpoint checkpointResources();
  bool commitResources(ResourceCheckpoint const& checkpoint);
  bool rollbackResources(ResourceCheckpoint const& checkpoint);

  // Clear colour for the next frames (the backdrop void colour).
  void setClearColour(float r, float g, float b, float a = 1.0F);

  // Direction is world-space, towards the distant light; ambient is 0..1.
  void setLighting(SceneLighting const& lighting);

  // Renders one frame. Windowed: presents to the swapchain. Offscreen:
  // renders into an internal image available through readback().
  bool draw(Camera const& camera,
            std::vector<DrawItem> const& items,
            std::vector<RendererVertex> const& depthLines,
            std::vector<RendererVertex> const& overlayLines);

  // Renders one frame into an arbitrary colour view (no present; blocks until
  // the target contents are complete; the target image ends in
  // COLOR_ATTACHMENT_OPTIMAL). `view` is the world->view matrix the blend
  // sort transforms through; `viewProj` is proj * view in the renderer's
  // conventions (cameraViewProjection, or mul(cameraProjectionFromTangents(...),
  // cameraViewFromPose(...)) for per-eye VR views). The target format must be
  // the `colourFormat` passed to initExternal.
  bool drawToView(VkImageView target, VkImage targetImage, Mat4 const& view,
                  Mat4 const& viewProj, Vec3 cameraPos,
                  std::vector<DrawItem> const& items,
                  std::vector<RendererVertex> const& depthLines,
                  std::vector<RendererVertex> const& overlayLines);

  // External-device path for the XR shell: the caller owns instance, physical
  // device and device (shutdown() releases only this renderer's resources).
  // `config.width/height` size the shared depth buffer; `colourFormat` must
  // match the views later passed to drawToView. Config::windowed is ignored.
  bool initExternal(Config const& config, VkInstance instance, VkPhysicalDevice physical,
                    VkDevice device, std::uint32_t queueFamily, VkFormat colourFormat);

  // Offscreen only: RGBA8 pixels (width * height * 4) of the last drawn frame.
  bool readback(std::vector<std::uint8_t>& pixelsOut);

  std::uint32_t width() const;
  std::uint32_t height() const;
  char const* deviceName() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  // Shared device resources (command/fence, shaders, descriptors, depth image,
  // pipelines) for init() and initExternal().
  bool createCommonResources(char const* shaderDir, VkFormat colourFormat);
};

} // namespace vredit
