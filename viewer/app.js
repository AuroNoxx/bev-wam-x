import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
const $=id=>document.getElementById(id), box=$('viewport');
const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance'});
renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));box.appendChild(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color('#0b1421');
const camera=new THREE.PerspectiveCamera(55,1,.1,800);camera.up.set(0,0,1);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.13;
controls.minDistance=3;controls.maxDistance=240;
scene.add(new THREE.HemisphereLight(0xffffff,0x526078,2.6));
const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(-40,-20,80);scene.add(light);
const grid=new THREE.GridHelper(240,48,0x35536b,0x203144);grid.rotation.x=Math.PI/2;grid.material.transparent=true;grid.material.opacity=.5;scene.add(grid);
const material=new THREE.MeshLambertMaterial();let voxels=null,capacity=0;
const dummy=new THREE.Object3D(),color=new THREE.Color();
const ego=new THREE.Mesh(new THREE.BoxGeometry(1,1,1),new THREE.MeshStandardMaterial({color:0x48e5d0,roughness:.55}));scene.add(ego);
const nose=new THREE.Mesh(new THREE.ConeGeometry(.55,1.8,3),new THREE.MeshBasicMaterial({color:0xe4fff7}));nose.rotation.z=-Math.PI/2;ego.add(nose);
const laneGeometry=new THREE.BufferGeometry(), laneMaterial=new THREE.PointsMaterial({color:0xa3b4bd,size:.16,sizeAttenuation:true,transparent:true,opacity:.65});
scene.add(new THREE.Points(laneGeometry,laneMaterial));
const actorGroup=new THREE.Group();scene.add(actorGroup);
const edgeGeometry=new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1)),edgeMaterial=new THREE.LineBasicMaterial({color:0xedf4fa,transparent:true,opacity:.65});
let actors=[],tracking=true,initialized=false,lastEgo=new THREE.Vector3(),lastFrame=-1,lastArrival=0,paused=false;
let renderCount=0,renderStart=performance.now(),renderFPS=0,updates=0,updateStart=performance.now(),occFPS=0;
function resize(){renderer.setSize(box.clientWidth,box.clientHeight);camera.aspect=box.clientWidth/box.clientHeight;camera.updateProjectionMatrix();}
new ResizeObserver(resize).observe(box);resize();
function reset(){controls.target.copy(lastEgo);camera.position.copy(lastEgo).add(new THREE.Vector3(-28,-36,30));controls.update();}
function modeButtons(a,b,first){$(a).classList.toggle('active',first);$(b).classList.toggle('active',!first);$(a).setAttribute('aria-pressed',String(first));$(b).setAttribute('aria-pressed',String(!first));}
async function command(action){try{const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});if(!r.ok)throw Error(`请求失败 ${r.status}`);$('error').textContent='';}catch(e){$('error').textContent=e.message;}}
$('car-follow').onclick=()=>command('follow');$('car-free').onclick=()=>command('free');
$('occ-follow').onclick=()=>{tracking=true;reset();modeButtons('occ-follow','occ-free',true);};
$('occ-free').onclick=()=>{tracking=false;modeButtons('occ-follow','occ-free',false);};
$('reset').onclick=reset;
$('top').onclick=()=>{controls.target.copy(lastEgo);camera.position.copy(lastEgo).add(new THREE.Vector3(0,-.01,75));controls.update();};
$('pause').onclick=async()=>{paused=!paused;await command(paused?'pause_display':'resume_display');$('pause').textContent=paused?'恢复 OCC 更新':'暂停 OCC 更新';};
function tint(tag){if([1,2,24,25,27].includes(tag))return 0x5a91aa;if([14,15,16,17,18,19].includes(tag))return 0xffab59;if([9,10].includes(tag))return 0x579c72;if([12,13].includes(tag))return 0xe7779e;return 0xa6a9bd;}
function update(data){
 if(!data.voxels||!data.ego)return;
 if(data.frame===lastFrame)return;lastFrame=data.frame;lastArrival=performance.now();updates++;
 $('connection').textContent='● 实时连接';$('topline').textContent=`${data.map}  /  FRAME ${data.frame}  /  表面占用`;
 $('speed').textContent=data.speed_kmh;$('control').textContent=`${data.throttle.toFixed(2)} / ${data.brake.toFixed(2)} / ${data.steer.toFixed(2)}`;
 $('hz').textContent=`${data.sim_hz} / ${data.target_sim_hz} Hz`;$('count').textContent=data.voxel_count.toLocaleString();$('voxel').textContent=`${data.voxel_size} m`;$('latency').textContent=`${data.processing_ms} ms`;
 modeButtons('car-follow','car-free',data.camera_mode==='follow');
 const p=new THREE.Vector3(...data.ego),delta=p.clone().sub(lastEgo);
 if(initialized&&tracking){camera.position.add(delta);controls.target.add(delta);}
 lastEgo.copy(p);if(!initialized){initialized=true;reset();}
 ego.position.copy(p);ego.position.z+=data.ego_extent[2];ego.rotation.z=data.yaw;ego.scale.set(data.ego_extent[0]*2,data.ego_extent[1]*2,data.ego_extent[2]*2);nose.position.set(.65,0,.6);
 grid.position.set(Math.round(p.x/5)*5,Math.round(p.y/5)*5,p.z-.05);
 const n=data.voxels.length;
 if(n>capacity){if(voxels){scene.remove(voxels);voxels.geometry.dispose();voxels.dispose();}capacity=Math.max(8192,2**Math.ceil(Math.log2(n)));voxels=new THREE.InstancedMesh(new THREE.BoxGeometry(1,1,1),material,capacity);voxels.instanceMatrix.setUsage(THREE.DynamicDrawUsage);voxels.frustumCulled=false;scene.add(voxels);}
 if(voxels){voxels.count=n;dummy.scale.setScalar(data.voxel_size*.93);dummy.rotation.set(0,0,0);for(let i=0;i<n;i++){const v=data.voxels[i];dummy.position.set(v[0],v[1],v[2]);dummy.updateMatrix();voxels.setMatrixAt(i,dummy.matrix);voxels.setColorAt(i,color.setHex(tint(v[3])));}voxels.instanceMatrix.needsUpdate=true;if(voxels.instanceColor)voxels.instanceColor.needsUpdate=true;}
 laneGeometry.setAttribute('position',new THREE.Float32BufferAttribute(data.lanes.flat(),3));laneGeometry.computeBoundingSphere();
 while(actors.length<data.actors.length){const a=new THREE.LineSegments(edgeGeometry,edgeMaterial);actors.push(a);actorGroup.add(a);}
 actors.forEach((a,i)=>{a.visible=i<data.actors.length;if(a.visible){const d=data.actors[i];a.position.set(d.p[0],d.p[1],d.p[2]+d.size[2]/2);a.scale.set(...d.size);a.rotation.z=d.yaw;}});
}
async function poll(){try{const r=await fetch('/api/frame',{cache:'no-store'});if(!r.ok)throw Error(r.status);update(await r.json());$('error').textContent='';}catch(e){$('connection').textContent='连接已断开';$('error').textContent='本地采集服务不可达：'+e.message;}finally{setTimeout(poll,150);}}
poll();
let lastRender=0;
function animate(now){requestAnimationFrame(animate);if(now-lastRender<1000/60)return;lastRender=now-(now-lastRender)%(1000/60);controls.update();renderer.render(scene,camera);renderCount++;if(now-renderStart>=1000){renderFPS=renderCount*1000/(now-renderStart);renderCount=0;renderStart=now;}if(now-updateStart>=2000){occFPS=updates*1000/(now-updateStart);updates=0;updateStart=now;}$('fps').textContent=`${occFPS.toFixed(1)} / ${renderFPS.toFixed(0)} FPS`;if(initialized&&now-lastArrival>3000)$('connection').textContent=paused?'OCC 已暂停':'数据未更新';}
requestAnimationFrame(animate);
window.addEventListener('error',event=>{$('error').textContent=event.message;});
