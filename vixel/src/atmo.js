"use strict";

const glsl = require("glslify");
const createCube = require("primitive-cube");
const unindex = require("unindex-mesh");
const renderEnvMap = require("regl-render-envmap");

module.exports = function createAtmosphereRenderer(regl) {
  const cube = unindex(createCube(1));

  const envmapCommand = regl({
    vert: require("./ver.glsl"),
    frag: require("./frag.glsl"),
    attributes: {
      position: cube,
    },
    uniforms: {
      sundir: regl.prop("sundir"),
      view: regl.prop("view"),
      projection: regl.prop("projection"),
    },
    viewport: regl.prop("viewport"),
    framebuffer: regl.prop("framebuffer"),
    count: cube.length / 3,
  });

  function render(opts) {
    opts = opts || {};
    opts.sunDirection =
      opts.sunDirection === undefined ? [0, 0.25, -1] : opts.sunDirection;
    opts.resolution = opts.resolution === undefined ? 1024 : opts.resolution;

    function renderer(config) {
      regl.clear({
        color: [0, 0, 0, 1],
        depth: 1,
        framebuffer: config.framebuffer,
      });
      envmapCommand({
        view: config.view,
        projection: config.projection,
        viewport: config.viewport,
        framebuffer: config.framebuffer,
        sundir: opts.sunDirection,
      });
    }

    return renderEnvMap(regl, renderer, {
      resolution: opts.resolution,
      cubeFBO: opts.cubeFBO,
    });
  }

  return render;
};
