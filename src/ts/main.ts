import { GPU, Input } from "gpu.js";
import readVox from "vox-reader";
import axios from "axios";
const voxF = require("../assets/roomSim.vox");

import Vixel from "../../vixel";
(async function() {
  const content = await axios.get(voxF, {
    responseType: "arraybuffer",
  });
  // await fs.mkdir('dir');
  // await fs.writeFile('dir/file.txt', 'hello world');
  start(readVox(Buffer.from(content.data, "binary")));
})();
const gpu = new GPU();
const rangeSlider = document.getElementById("depth") as HTMLInputElement;
const DENSITY_AIR = 1.225;
const K_AIR = 144120.0;
const simpleRender =
  window.location.hash.includes("simple") ||
  !window.location.hash.includes("fancy");
const fancyRender =
  window.location.hash.includes("fancy") ||
  !window.location.hash.includes("simple");
function start(data) {
  console.log(data, data.size, data.matl, data.imap);
  rangeSlider.min = "0";
  rangeSlider.max = data.size.y - 1 + "";
  const speakers = [{ x: 60, y: 30, z: 10 }];
  rangeSlider.valueAsNumber = speakers[0].y;
  let oldV = Math.floor(rangeSlider.valueAsNumber + 0);
  let units_per_meter = 10.0;
  let ticks_per_second = 44100.0;
  let R = units_per_meter / ticks_per_second;
  const size = data.size;

  const vcanvas = document.createElement("canvas");
  document.body.appendChild(vcanvas);

  // Create a vixel object with a canvas and a width (x), height (y), and depth (z).
  const vixel = new Vixel(vcanvas, size.x, size.z, size.y);
  vixel.sun(1001.5, 11, 0.1, 1);
  vixel.camera(
    [size.x * 0.5, size.z * 0.5, size.y * 2], // Camera position
    [size.x / 2, size.z / 2, size.y / 2], // Camera target
    [0, 1, 0], // Up
    Math.PI / 4 // Field of view
  );
  const solveScottA = gpu
    .createKernel(
      `function (
      a,
      b,
      c,
      d,
      R
    ) {
      let cB =
        (R / c[this.thread.z][this.thread.y][this.thread.x]) *
        b[this.thread.z][this.thread.y][this.thread.x];
      let nR = cB + a[this.thread.z][this.thread.y][this.thread.x]; //cA + dt * (diffA * lapA - cA * cB * cB + f * (1 - cA));
     
      return nR * 0.999;
    }`
    )
    // .setPipeline(true)
    // .setImmutable(true)
    .setOutput([size.z, size.y, size.x]);
  const solveScottB = gpu
    .createKernel(
      `function(
      a,
      b,
      c,
      d,
      R
    ) {
      let lapA = 0;
      for (let i = -1; i <= 1; i++) {
        for (let j = -1; j <= 1; j++) {
          for (let k = -1; k <= 1; k++) {
            if (Math.abs(i) + Math.abs(j) + Math.abs(k) === 1) {
              let X = Math.min(
                this.constants.size_x - 1,
                Math.max(0, this.thread.z + i)
              );
              let Y = Math.min(
                this.constants.size_y - 1,
                Math.max(0, this.thread.y + j)
              );
              let Z = Math.min(
                this.constants.size_z - 1,
                Math.max(0, this.thread.x + k)
              );
              if (
                X !== this.thread.z + i ||
                Y !== this.thread.y + j ||
                Z !== this.thread.x + k
              ) {
                lapA += 0;
              } else {
                lapA +=
                  a[X][Y][Z] - a[this.thread.z][this.thread.y][this.thread.x];
              }
            }
          }
        }
      }
      let cB = b[this.thread.z][this.thread.y][this.thread.x];
      let nR = cB + R * d[this.thread.z][this.thread.y][this.thread.x] * lapA; //cA + dt * (diffA * lapA - cA * cB * cB + f * (1 - cA));
      
      return nR;
    }`
    )
    .setConstants({ size_x: size.x, size_y: size.y, size_z: size.z })
    // .setPipeline(true)
    // .setImmutable(true)
    .setOutput([size.z, size.y, size.x]);
  let DENSITY: any | number[][][] = new Array(size.x)
    .fill(0)
    .map((x) =>
      new Array(size.y)
        .fill(0)
        .map((y) => new Array(size.z).fill(0).map((z) => DENSITY_AIR))
    );
  let K: any | number[][][] = new Array(size.x)
    .fill(0)
    .map((x) =>
      new Array(size.y)
        .fill(0)
        .map((y) => new Array(size.z).fill(0).map((z) => K_AIR))
    );
  let POSITION: any | number[][][] = new Array(size.x)
    .fill(0)
    .map((x) =>
      new Array(size.y)
        .fill(0)
        .map((y) => new Array(size.z).fill(0).map((z) => 0))
    );
  let VELOCITY: any | number[][][] = new Array(size.x)
    .fill(0)
    .map((x) =>
      new Array(size.y)
        .fill(0)
        .map((y) => new Array(size.z).fill(0).map((z) => 0))
    );
  let soundVoxels = [];
  for (let j = 0; j < data.xyzi.values.length; j++) {
    let xyziT = data.xyzi.values[j];
    DENSITY[xyziT.x][xyziT.y][xyziT.z] = 1000;
    K[xyziT.x][xyziT.y][xyziT.z] = 144120.0; //2186780917.0;
    let emit = undefined;
    if (xyziT.i != 249) {
      emit = 1;
      // console.log(xyziT.i)
    }
    if (xyziT.y > 1) {
      vixel.set(xyziT.x, xyziT.z, size.y - xyziT.y - 1, {
        red: data.rgba.values[xyziT.i % 256].r / 255, // Red component
        green: data.rgba.values[xyziT.i % 256].g / 255, // Green component
        blue: data.rgba.values[xyziT.i % 256].b / 255, // Blue component,
        emit: emit,
      });
    }
  }
  // vixel.sun(1000, 10, 0.1,1);
  // Take 1024 path traced samples per pixel
  vixel.sample(8);

  // Show the result on the canvas
  vixel.display();

  POSITION[speakers[0].x][speakers[0].y][speakers[0].z] = 1;
  // DENSITY=new Input(DENSITY,[128,128,128]);
  // K=new Input(K,[128,128,128]);
  // for (let j = 60; j < 69; j++) {
  //   for (let k = 60; k < 69; k++) {
  //     for (let l = 60; l < 69; l++) {
  //       A[j][k][l] = 0;
  //       B[j][k][l] = Math.random();
  //     }
  //   }
  // }

  const render = gpu
    .createKernel(
      `function (a, b, id) {
      let sm = 0;
      let dst = 128;
      let idx = Math.floor(id);//(tim / 1000 * 128) % 127.5);
      for (let i = 1; i < this.constants.size_y; i++) {
        let ai = a[this.thread.x][i][this.thread.y];
        let bi = b[this.thread.x][i][this.thread.y];
        // let rat = bi / (ai + bi);
        let rat = ai / 1000;
        if(i==idx){
        sm = Math.max(sm, Math.abs(bi) );
        }
        //sm += rat;
        if (rat >= 0.5) {
          dst = Math.min(dst - rat, i);
          break;
        }
      }
      let I = sm / 2;
      let O=0.0;
      if(dst>=idx){
        O=0.5;
      }
      
      // let aix = a[this.thread.x][this.thread.y][idx];
      // let bix = b[this.thread.x][this.thread.y][idx];
      // // this.color(bix/(aix+bix)>0.5?1:0,bix/(aix+bix)>0.5?1:0,bix/(aix+bix)>0.5?1:0, 1);
      // this.color(bix / (aix + bix), aix, bix, 1);
      this.color(
        (dst / this.constants.size_y) * (1 - I) + I,
        (dst / this.constants.size_y) * (1 - I),
        (dst / this.constants.size_y) * (1 - I)+O,
        1
      );
    }`
    )
    .setConstants({ size_x: size.x, size_y: size.y, size_z: size.z })
    .setOutput([size.x, size.z])
    .setGraphical(true);

  // render(A,B);

  const canvas = render.canvas;
  const c2 = document.getElementsByTagName("body")[0].appendChild(canvas);
  c2.width = 128;
  c2.height = 128;
  (window as any).v = 0;
  function update() {
    for (let q = 0; q < 1; q++) {
      POSITION[speakers[0].x][speakers[0].y][speakers[0].z] =
        Math.sin(((window as any).v / 44100) * 440 * Math.PI * 2) *
        Math.max(1 - (((window as any).v / 44100) * 440) / 1, 0);
      const nA = solveScottA(POSITION, VELOCITY, DENSITY, K, R);

      POSITION.delete && POSITION.delete();
      POSITION = nA;

      const nB = solveScottB(POSITION, VELOCITY, DENSITY, K, R);

      VELOCITY.delete && VELOCITY.delete();
      VELOCITY = nB;

      (window as any).v += 1;
    }
    // console.log((A as any).toArray()[0][0][0]);
    // console.log((B as any).toArray()[0][0][0]);
    // ;
    requestAnimationFrame(update);
  }
  const gogo = () => {
    let vo = Math.floor(rangeSlider.valueAsNumber);
    document.getElementById("depth-disp").innerText = vo + "";
    if (fancyRender) {
      if (oldV !== vo || soundVoxels.length === 0) {
        oldV = vo;
        for (let i = 0; i < soundVoxels.length; i++) {
          let xyziT = soundVoxels[i];
          // vixel.unset(
          //   xyziT.x,
          //   xyziT.z,
          //   size.y - xyziT.y - 1
          //   // {
          //   //   red: 1,
          //   //   green:0,
          //   //   blue:0,
          //   //   alpha:0.5
          //   // }
          // );
          vixel.set(xyziT.x, xyziT.z, size.y - xyziT.y - 1, {
            red: 1,
            green: 1,
            blue: 1,
            transparent:
              1,
          });
        }

        soundVoxels = [];

        for (let i = 0; i < size.x; i++) {
          for (let j = 0; j < size.z; j++) {
            let xyziT = { x: i, y: Math.floor(vo), z: j };
            if (DENSITY[xyziT.x][xyziT.y][xyziT.z] < 500) {
              soundVoxels.push(xyziT);
            }
          }
        }
      }
      for (let i = 0; i < soundVoxels.length; i++) {
        let xyziT = soundVoxels[i];

        let VQ = VELOCITY[xyziT.x][xyziT.y][xyziT.z];

        let PQ = POSITION[xyziT.x][xyziT.y][xyziT.z];
        let PSD =
          ((R * R) / DENSITY[xyziT.x][xyziT.y][xyziT.z]) *
          K[xyziT.x][xyziT.y][xyziT.z];
        let Q =
          ((Math.sign(VQ) * Math.sqrt((1 / PSD) * VQ * VQ + PQ * PQ)) / 44100) *
          440; //VELOCITY[xyziT.x][xyziT.y][xyziT.z];
        vixel.set(xyziT.x, xyziT.z, size.y - xyziT.y - 1, {
          red: Math.abs(Q) > 1 / 8 ? 1 / 2 + Math.sign(Q) / 2 : 1,
          green: Math.abs(Q) > 1 / 8 ? 1 / 2 - Math.sign(Q) / 2 : 1,
          blue: Math.abs(Q) > 1 / 8 ? 1 / 2 - Math.sign(Q) / 2 : 1,
          transparent:
            Math.abs(Q) > 1 / 8
              ? (1 - Math.floor(Math.abs(Q) * 8) / 8) / 2 + 0.5
              : 1,
        });
      }

      // vixel.sun(1000, 10, 0.1,1);
      // Take 1024 path traced samples per pixel
      vixel.sample(64);

      // Show the result on the canvas
      vixel.display();
    }
    if (simpleRender) {
      render(DENSITY, VELOCITY, vo);
    }

    c2.width = size.x;
    c2.height = size.z;

    vcanvas.width = size.x * 4;
    vcanvas.height = size.z * 4;
    setTimeout(gogo, 1000);
  };
  gogo();
  function exportToJsonFile(jsonData) {
    let dataStr = JSON.stringify(jsonData);
    let dataUri =
      "data:application/json;charset=utf-8," + encodeURIComponent(dataStr);

    let exportFileDefaultName = "data.json";

    let linkElement = document.createElement("a");
    linkElement.setAttribute("href", dataUri);
    linkElement.setAttribute("download", exportFileDefaultName);
    linkElement.click();
  }
  // (window as any).getV = () => {
  //   let aA: Float32Array[][] = (A as any).toArray();
  //   let aB: Float32Array[][] = (B as any).toArray();
  //   exportToJsonFile({ A: aA.map(x => [...x.map(r => [...r])]), B: aB.map(x => [...x.map(r => [...r])]) });

  // }
  (window as any).u = update;
  update();
}
