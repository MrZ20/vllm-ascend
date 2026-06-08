import fs from 'fs';
const ROOT='d:/57108/Desktop/code/vllm-ascend/vllm_ascend';
const {nodes,edges}=JSON.parse(fs.readFileSync(ROOT+'/.understand-anything/tmp/ua-15-full.json','utf8'));
const input=JSON.parse(fs.readFileSync(ROOT+'/.understand-anything/tmp/ua-file-analyzer-input-15.json','utf8'));
const files=input.batchFiles.map(f=>f.path).sort();
const parts=2;
const sz=Math.ceil(files.length/parts);
const groups=[];
for(let i=0;i<parts;i++)groups.push(new Set(files.slice(i*sz,(i+1)*sz)));
// node filePath -> group index
function gi(fp){for(let i=0;i<parts;i++)if(groups[i].has(fp))return i;return 0;}
const nodeGroup=new Map();
nodes.forEach(n=>nodeGroup.set(n.id, gi(n.filePath)));
for(let i=0;i<parts;i++){
  const pn=nodes.filter(n=>nodeGroup.get(n.id)===i);
  const ids=new Set(pn.map(n=>n.id));
  const pe=edges.filter(e=>ids.has(e.source));
  const out={nodes:pn,edges:pe};
  fs.writeFileSync(ROOT+'/.understand-anything/intermediate/batch-15-part-'+(i+1)+'.json',JSON.stringify(out,null,1));
  console.log('part',i+1,'nodes',pn.length,'edges',pe.length);
  // validate edges: source must be in pn
  pe.forEach(e=>{if(!ids.has(e.source))console.log('BAD source',e.source);});
}
