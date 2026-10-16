#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "camera.h"
#include "geometry.h"

namespace vredit {

// Vulkan viewport renderer: draws cubes per DrawItem, depth-tested lines
// (floor grid) and always-on-top overlay lines (gizmo handles). Vulkan 1.3
// with dynamic rendering; no Vulkan types leak past this header.
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

  // Renders one frame. Windowed: presents to the swapchain. Offscreen:
  // renders into an internal image available through readback().
  bool draw(Camera const& camera,
            std::vector<DrawItem> const& items,
            std::vector<RendererVertex> const& depthLines,
            std::vector<RendererVertex> const& overlayLines);

  // Offscreen only: RGBA8 pixels (width * height * 4) of the last drawn frame.
  bool readback(std::vector<std::uint8_t>& pixelsOut);

  std::uint32_t width() const;
  std::uint32_t height() const;
  char const* deviceName() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace vredit
