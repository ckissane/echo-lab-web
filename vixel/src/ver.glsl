
      precision highp float;

      attribute vec3 position;

      uniform mat4 view, projection;

      varying vec3 pos;

      void main() {
        gl_Position = projection * view * vec4(position, 1);
        pos = position;
      }