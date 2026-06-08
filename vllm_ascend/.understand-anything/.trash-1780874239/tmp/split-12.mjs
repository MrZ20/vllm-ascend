import fs from 'fs';
const g=JSON.parse(fs.readFileSync('d:/57108/Desktop/code/vllm-ascend/vllm_ascend/.understand-anything/tmp/gen-12-out.json','utf8'));
const inp=JSON.parse(fs.readFileSync('d:/57108/Desktop/code/vllm-ascend/vllm_ascend/.understand-anything/tmp/ua-file-analyzer-input-12.json','utf8'));
const files=inp.batchFiles.map(f=>f.path).sort();
const parts=2;
const sz=Math.ceil(files.length/parts);
const groups=[];
for(let i=0;i<parts;i++) groups.push(new Set(files.slice(i*sz,(i+1)*sz)));

function pathOf(node){return node.filePath;}
for(let k=0;k<parts;k++){
  const set=groups[k];
  const pn=g.nodes.filter(n=>set.has(pathOf(n)));
  const ids=new Set(pn.map(n=>n.id));
  const pe=g.edges.filter(e=>ids.has(e.source));
  fs.writeFileSync(`d:/57108/Desktop/code/vllm-ascend/vllm_ascend/.understand-anything/intermediate/batch-12-part-${k+1}.json`,JSON.stringify({nodes:pn,edges:pe},null,1));
  console.log('part',k+1,'nodes',pn.length,'edges',pe.length);
}
