import { GPU } from 'gpu.js';
import readVox from 'vox-reader';
import axios from 'axios';
const voxF = require('../assets/room.vox');
(async function () {
  const content = await axios.get(voxF, {
    responseType: 'arraybuffer'
  });
  // await fs.mkdir('dir');
  // await fs.writeFile('dir/file.txt', 'hello world');
  start(readVox(Buffer.from(content.data, 'binary')));

})();
const gpu = new GPU();
const solveScottA = gpu.createKernel(function (a: number[][][], b: number[][][], c: number[][][], d: number[][][], R:number) {

  let cB = R/c[this.thread.z][this.thread.y][this.thread.x]*b[this.thread.z][this.thread.y][this.thread.x];
  let nR = cB+a[this.thread.z][this.thread.y][this.thread.x];//cA + dt * (diffA * lapA - cA * cB * cB + f * (1 - cA));
  // let X = this.thread.z - 128 / 2;
  // let Y = this.thread.y - 128 / 2;
  // let Z = this.thread.x - 128 / 2;

  // if (Math.sqrt((X)**2+Y**2+0*Z**2)>128/2-4) {
  //   nR = fB;
  // }
  // if (Math.sqrt((X)**2+Y**2)<128/4) {
  //   nR =0.5;
  // }
  return nR*0.999;
}).setPipeline(true).setImmutable(true).setOutput([128, 128, 128]);
const flipXZ = gpu.createKernel(function (a: number[][][]) {
  return a[this.thread.z][this.thread.y][this.thread.x];
}).setPipeline(true).setOutput([128, 128, 128]);
const solveScottB = gpu.createKernel(function (a: number[][][], b: number[][][],c: number[][][], d: number[][][],R:number) {
  
  let lapA = 0;
  for (let i = -1; i <= 1; i++) {
    for (let j = -1; j <= 1; j++) {
      for (let k = -1; k <= 1; k++) {
        if (Math.abs(i) + Math.abs(j) + Math.abs(k) === 1) {
          let X = Math.min(128 - 1, Math.max(0, this.thread.z + i));
          let Y = Math.min(128 - 1, Math.max(0, this.thread.y + j));
          let Z = Math.min(128 - 1, Math.max(0, this.thread.x + k));
          if (X !== this.thread.z + i || Y !== this.thread.y + j || Z !== this.thread.x + k) {
            lapA += 0;
          } else {
            lapA += a[X][Y][Z]-a[this.thread.z][this.thread.y][this.thread.x];
          }
        }
      }
    }
  }
  // lapA -= a[this.thread.z][this.thread.y][this.thread.x] * 6;
  // lapA *= 1 / 6;
  // let cA = a[this.thread.z][this.thread.y][this.thread.x];
  let cB = b[this.thread.z][this.thread.y][this.thread.x];
  let nR = cB+R*d[this.thread.z][this.thread.y][this.thread.x]*lapA;//cA + dt * (diffA * lapA - cA * cB * cB + f * (1 - cA));
  // let X = this.thread.z - 128 / 2;
  // let Y = this.thread.y - 128 / 2;
  // let Z = this.thread.x - 128 / 2;

  // if (Math.sqrt((X)**2+Y**2+0*Z**2)>128/2-4) {
  //   nR = fB;
  // }
  // if (Math.sqrt((X)**2+Y**2)<128/4) {
  //   nR =0.5;
  // }
  return nR;
}).setPipeline(true).setImmutable(true).setOutput([128, 128, 128]);
const DENSITY_AIR=1.225;
const K_AIR = 144120.0;
function start(data) {
  console.log(data,data.size)
  const speakers = [{ x: 60, y: 52, z: 10 }];
  let units_per_meter = 8.0;
  let ticks_per_second = 44100.0;
  let R = units_per_meter / ticks_per_second;
  const size = data.size;
  let DENSITY: any | number[][][] = new Array(128).fill(0).map(x => new Array(128).fill(0).map(y => new Array(128).fill(0).map(z => DENSITY_AIR)));
  let K: any | number[][][] = new Array(128).fill(0).map(x => new Array(128).fill(0).map(y => new Array(128).fill(0).map(z => K_AIR)));
  let POSITION: any | number[][][] = new Array(128).fill(0).map(x => new Array(128).fill(0).map(y => new Array(128).fill(0).map(z => 0)));
  let VELOCITY: any | number[][][] = new Array(128).fill(0).map(x => new Array(128).fill(0).map(y => new Array(128).fill(0).map(z => 0)));
  for (let j = 0; j < data.xyzi.values.length; j++){
    let xyziT = data.xyzi.values[j];
    DENSITY[xyziT.x][xyziT.y][xyziT.z] = 1000;
    K[xyziT.x][xyziT.y][xyziT.z] = 144120.0;//2186780917.0;
  }

  POSITION[speakers[0].x][speakers[0].y][speakers[0].z] = 1;
  // for (let j = 60; j < 69; j++) {
  //   for (let k = 60; k < 69; k++) {
  //     for (let l = 60; l < 69; l++) {
  //       A[j][k][l] = 0;
  //       B[j][k][l] = Math.random();
  //     }
  //   }
  // }

  const render = gpu.createKernel(function (a: number[][][], b: number[][][], tim: number) {
    let sm = 0;
    let dst = 128;
    
    for (let i = 0; i < 128; i++) {
      let ai = a[this.thread.x][i][this.thread.y];
      let bi = b[this.thread.x][i][this.thread.y];
      // let rat = bi / (ai + bi);
      let rat = ai / 1000;
      sm = Math.max(sm,Math.abs(bi)*(1-sm));
      //sm += rat;
      if (rat >= 0.5) {
        dst = Math.min(dst - rat, i);
        break;
      }
    }
    let I = sm/2;
    // let idx = Math.floor(128 / 2);//(tim / 1000 * 128) % 127.5);
    // let aix = a[this.thread.x][this.thread.y][idx];
    // let bix = b[this.thread.x][this.thread.y][idx];
    // // this.color(bix/(aix+bix)>0.5?1:0,bix/(aix+bix)>0.5?1:0,bix/(aix+bix)>0.5?1:0, 1);
    // this.color(bix / (aix + bix), aix, bix, 1);
    this.color(dst / 128*(1-I)+I, dst / 128*(1-I), dst / 128*(1-I), 1);
  })
    .setOutput([128, 128])
    .setGraphical(true);

  // render(A,B);

  const canvas = render.canvas;
  const c2=document.getElementsByTagName('body')[0].appendChild(canvas);
  c2.width = 128;
  c2.height = 128;
  (window as any).v = 0;
  function update() {
    for (let q = 0; q < 1; q++) {
      const nA = solveScottA(POSITION,VELOCITY,DENSITY,K,R);
      const nB = solveScottB(POSITION,VELOCITY,DENSITY,K,R);
      POSITION.delete && POSITION.delete();
      VELOCITY.delete && VELOCITY.delete();
      POSITION = nA;
      VELOCITY = nB;
    }
    // console.log((A as any).toArray()[0][0][0]);
    // console.log((B as any).toArray()[0][0][0]);
    // ;
    (window as any).v += 1;
    requestAnimationFrame(update);
  }
  window.setInterval(() => {
    render(DENSITY, POSITION, new Date().getTime())
    c2.width = 128;
  c2.height = 128;
  }, 1000/4);
  function exportToJsonFile(jsonData) {
    let dataStr = JSON.stringify(jsonData);
    let dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);

    let exportFileDefaultName = 'data.json';

    let linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
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