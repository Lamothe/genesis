#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "camera.h"
#include "geometry.h"
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
  void shutdown();

  bool resize(std::uint32_t width, std::uint32_t height);

  // Registers static mesh data and returns a handle (0 is the built-in cube).
  // Safe to call at any time; internally synchronises with in-flight frames.
  MeshHandle addMesh(MeshData const& data);

  // Registers an RGBA8 texture and returns a handle (0 is the built-in white
  // fallback). Up to 256 textures; safe to call at any time.
  TextureHandle addTexture(Image const& image, bool repeatU, bool repeatV);

  // Clear colour for the next frames (the backdrop void colour).
  void setClearColour(float r, float g, float b, float a = 1.0F);

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
