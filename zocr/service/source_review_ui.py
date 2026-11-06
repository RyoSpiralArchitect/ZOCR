HTML = r'''<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ZOCR 原文確認・学習同意</title>
<style>
body{font:16px system-ui;margin:1.5rem;background:#f6f7f9;color:#202530}main{max-width:1100px;margin:auto}section{background:white;padding:1rem;margin:1rem 0;border:1px solid #ccd;border-radius:6px}label{display:block;margin:.6rem 0}input,textarea,select,button{font:inherit;padding:.4rem}input[type=text],textarea{width:95%}textarea{min-height:5rem}button{margin:.3rem}#viewport{overflow:auto;max-height:450px;border:1px solid #aaa}canvas{display:block;touch-action:none}#status{white-space:pre-wrap}small{display:block;color:#526}#glyphs{white-space:pre-wrap}.warning{background:#fff4d8;padding:.7rem}
</style><main><h1>原文確認と学習素材の登録</h1>
<p class="warning">機械の予測・PDFの文字層は正解データではありません。画像を人が確認してください。訂正をまとめて保存した後、明示的に同意した素材だけを登録します。登録後の訂正追加は、このレビューの既存素材を失効させます。本番モデルの自動更新は行いません。</p>
<section><label>APIキー（この画面内だけで使用。保存しません）<input id="key" type="password" autocomplete="off"></label>
<label>完了したジョブID<input id="job" type="text"></label><button id="create">原文確認を作成</button>
<label>既存の原文確認ID<input id="reviewId" type="text"></label><button id="load">確認を読み込む</button></section>
<section><button id="prev">前へ</button><span id="position"></span><button id="next">次へ</button>
<pre id="provenance"></pre><label>原画像上の訂正範囲 [x0,y0,x1,y1]<input id="bbox" type="text"></label><button id="crop">画像を表示・範囲を更新</button>
<label>表示倍率<input id="zoom" type="range" min="1" max="4" value="2" step=".5"></label>
<div id="viewport"><canvas id="canvas"></canvas></div><small>学習・候補選択用の字形は、画像をドラッグして範囲を指定し、文字を一つずつ入力してください。自動の文字ラベル付けは行いません。</small>
<label>選択した字形の文字（1文字）<input id="glyph" type="text"></label><button id="addGlyph">字形を追加</button><button id="clearGlyphs">字形をすべて解除</button><pre id="glyphs"></pre>
<label>人が画像で確認したテキスト<textarea id="text"></textarea></label><label>確認・訂正の理由<input id="reason" type="text"></label>
<label><input id="confirmed" type="checkbox"> 原文画像とテキスト・範囲を人が照合しました</label>
<label><input id="consent" type="checkbox"> この訂正と手動字形ラベルを、以下の目的で学習・評価に使うことに同意します</label>
<label>文書ファミリーID<input id="group" type="text" placeholder="同じ調達案件・関連文書群は同じID"></label>
<label>用途<select id="role"><option value="">選択してください</option><option value="train">学習</option><option value="selection">候補選択・開発確認</option><option value="test">未見の独立最終評価</option></select></label>
<label>出自<select id="kind"><option value="">選択してください</option><option value="real_document">実文書（操作者の申告）</option><option value="fixture">合成・試験用fixture</option></select></label>
<label><input id="unseen" type="checkbox"> 実文書の最終評価用：学習・候補選択・既存の性能確認に使っていない、別ファミリーの未見資料であると申告します</label>
<p>既に結果を見た資料は、最終評価には使わないでください。最終評価の文書ファミリーは一度だけ使用でき、中断した場合も消費済みとして扱います。</p>
<button id="draft">この領域の確認案を保持</button><span id="draftCount"></span></section>
<section><button id="save">1. 確認済みの訂正をまとめてサーバーに保存</button><button id="register" disabled>2. 同意した素材を登録</button><p id="status" role="status"></p></section></main>
<script>
const $=id=>document.getElementById(id), drafts=new Map();
let review=null,index=0,image=null,pending=null,start=null,saved=false,displayedBox=null,generation=0;
const status=text=>$('status').textContent=text;
function invalidate(){saved=false;$('register').disabled=true}
for(const id of ['text','reason','confirmed','consent','unseen','group','role','kind','bbox'])$(id).addEventListener('input',invalidate);
function secure(){if(!window.isSecureContext)throw Error('APIキーはHTTPSまたはループバック接続でのみ使用できます')}
async function api(path,body){secure();const response=await fetch(path,{method:body===undefined?'GET':'POST',headers:{'X-API-Key':$('key').value,'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body),cache:'no-store'});if(!response.ok){const value=await response.json();throw Error(value.detail||`HTTP ${response.status}`)}return response.json()}
function action(id,fn){$(id).onclick=async()=>{try{await fn()}catch(error){status(error.message)}}}
function current(){if(!review)throw Error('確認を読み込んでください');return review.result.events[index]}
function box(){const value=JSON.parse($('bbox').value);if(!Array.isArray(value)||value.length!==4||!value.every(Number.isFinite))throw Error('訂正範囲を4つの数値で指定してください');return value}
function draw(){const c=$('canvas'),ctx=c.getContext('2d');if(!image)return;c.width=image.width;c.height=image.height;c.style.width=`${image.width*Number($('zoom').value)}px`;c.style.height=`${image.height*Number($('zoom').value)}px`;ctx.drawImage(image,0,0);const rectangles=[...(drafts.get(current().event_id)?.glyphs||[]).map(g=>g.bbox),...(pending?[pending]:[])];ctx.strokeStyle='#d22';ctx.lineWidth=1;for(const b of rectangles)ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1])}
async function crop(){
secure();const event=current(),selected=box(),ticket=++generation;image=null;displayedBox=null;start=null;pending=null;
const response=await fetch(`/v1/reviews/${encodeURIComponent(review.review_id)}/events/${encodeURIComponent(event.event_id)}/image?bbox=${encodeURIComponent(JSON.stringify(selected))}`,{headers:{'X-API-Key':$('key').value},cache:'no-store'});
if(!response.ok)throw Error((await response.json()).detail||'画像を取得できません');
const url=URL.createObjectURL(await response.blob()),picture=new Image();try{await new Promise((resolve,reject)=>{picture.onload=resolve;picture.onerror=()=>reject(Error('画像を読み込めません'));picture.src=url});
if(ticket!==generation||current().event_id!==event.event_id||JSON.stringify(box())!==JSON.stringify(selected))return;
image=picture;displayedBox=JSON.stringify(selected);draw()}finally{URL.revokeObjectURL(url)}}
async function show(){const event=current(),evidence=event.evidence.a[0],draft=drafts.get(event.event_id);$('position').textContent=`${index+1} / ${review.result.events.length}`;$('provenance').textContent=JSON.stringify({source_kind:evidence.source_kind,source_sha256:evidence.source_sha256,source_page:evidence.source_page,record_kind:evidence.record_kind,observed_text:evidence.observed_text,confidence:evidence.confidence},null,2);$('bbox').value=JSON.stringify(draft?.bbox||evidence.bbox);$('text').value=draft?.text??evidence.observed_text??evidence.text??'';for(const id of ['reason','group','role','kind'])$(id).value=draft?.[id]||'';for(const id of ['confirmed','consent','unseen'])$(id).checked=Boolean(draft?.[id]);$('glyphs').textContent=JSON.stringify(draft?.glyphs||[],null,2);$('glyph').value='';image=null;await crop()}
function accept(value){if(value.result.mode!=='source_observations')throw Error('この画面は単一文書の原文確認用です');review=value;index=0;drafts.clear();saved=false;$('register').disabled=true;$('reviewId').value=review.review_id;return show()}
function retain(){const event=current(),existing=drafts.get(event.event_id);if($('confirmed').checked&&(!image||displayedBox!==JSON.stringify(box())))throw Error('訂正範囲の画像を更新し、人が照合してから確認してください');drafts.set(event.event_id,{text:$('text').value,reason:$('reason').value,confirmed:$('confirmed').checked,consent:$('consent').checked,unseen:$('unseen').checked,group:$('group').value.trim(),role:$('role').value,kind:$('kind').value,bbox:box(),glyphs:existing?.glyphs||[]});saved=false;$('register').disabled=true;$('draftCount').textContent=`保持中: ${drafts.size}領域`;return drafts.get(event.event_id)}
function point(event){const rect=$('canvas').getBoundingClientRect();return [Math.max(0,Math.min(image.width,Math.round((event.clientX-rect.left)*image.width/rect.width))),Math.max(0,Math.min(image.height,Math.round((event.clientY-rect.top)*image.height/rect.height)))]}
$('canvas').onpointerdown=event=>{if(!image)return;start=point(event);$('canvas').setPointerCapture(event.pointerId)};
$('canvas').onpointerup=event=>{if(!start)return;const end=point(event);pending=[Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])];start=null;draw()};
$('zoom').oninput=draw;
action('create',async()=>accept((await api('/v1/source-reviews',{job_id:$('job').value.trim()})).review));
action('load',async()=>accept((await api(`/v1/reviews/${encodeURIComponent($('reviewId').value.trim())}`)).review));
action('prev',async()=>{retain();index=Math.max(0,index-1);await show()});action('next',async()=>{retain();index=Math.min(review.result.events.length-1,index+1);await show()});
action('crop',async()=>{$('confirmed').checked=false;retain().glyphs=[];await crop();$('glyphs').textContent='[]'});
action('draft',async()=>{retain();status('確認案を画面内に保持しました。まだサーバーには保存していません。')});
action('addGlyph',async()=>{const char=$('glyph').value;if([...char].length!==1||!char.trim()||!pending||pending[0]>=pending[2]||pending[1]>=pending[3])throw Error('文字を1つ入力し、有効な字形範囲をドラッグしてください');const draft=retain();draft.glyphs.push({text:char,bbox:pending});pending=null;$('glyph').value='';$('glyphs').textContent=JSON.stringify(draft.glyphs,null,2);draw()});
action('clearGlyphs',async()=>{retain().glyphs=[];pending=null;$('glyphs').textContent='[]';draw()});
action('save',async()=>{retain();const decisions=[...drafts].filter(([id,d])=>d.confirmed).map(([id,d])=>{if(!d.reason.trim())throw Error('確認済みの各領域に理由を入力してください');return {event_id:id,status:'corrected',reason:d.reason,correction:{side:'a',text:d.text,bbox:d.bbox}}});if(!decisions.length||decisions.length>1000)throw Error('1〜1000領域を人が確認してください');review=(await api(`/v1/reviews/${review.review_id}/decisions`,{expected_revision:review.revision,submission_id:crypto.randomUUID(),source_hashes:review.result.review.source_hashes,decisions})).review;saved=true;$('register').disabled=false;status(`訂正をサーバーに保存しました（版${review.revision}）。登録前に、対象・用途・同意を確認してください。`)});
action('register',async()=>{if(!saved)throw Error('先に訂正をまとめて保存してください');const selected=[...drafts].filter(([id,d])=>d.confirmed&&d.consent);if(!selected.length)throw Error('学習・評価利用への明示的な同意が必要です');const samples=[];for(const [id,d] of selected){if(!d.group||!d.role||!d.kind)throw Error('文書ファミリー・用途・出自を明示してください');const result=await api(`/v1/reviews/${review.review_id}/learning-samples`,{expected_revision:review.revision,event_id:id,consent:true,role:d.role,source_kind:d.kind,document_group_id:d.group,final_test_attestation:d.unseen,glyphs:d.glyphs});samples.push(result.sample.object_id)}status(`登録済み（本番への自動反映なし）:\n${samples.join('\n')}`)});
</script></html>'''
