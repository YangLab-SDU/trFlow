/* Local trFlow workspace. No cloud requests; structures and state come from the local API. */
(() => {
  'use strict';
  const $ = (selector, node = document) => node.querySelector(selector);
  const $$ = (selector, node = document) => Array.from(node.querySelectorAll(selector));
  const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const i18n=window.TrFlowI18n;
  const t=(key,parameters={})=>i18n.t(key,parameters);
  const translated=(key,parameters={})=>escapeHTML(t(key,parameters));
  const label=key=>`<span data-i18n="${key}">${translated(key)}</span>`;
  const message=(key,parameters={})=>({key,parameters});
  class UIError extends Error {
    constructor(key,parameters={}){super(t(key,parameters));this.key=key;this.parameters=parameters;}
  }
  const knownMessageKey=key=>Object.prototype.hasOwnProperty.call(i18n.messages.en,key);
  const errorRaw=value=>value&&typeof value==='object'?(value.error??value.detail??value.message??''):value??'';
  function legacyServerMessage(raw){
    // Match only complete application messages, never a traceback or an exception prefix.
    if(/^(?:(?:[\w.]*?(?:Error|Exception|Warning))\s*:|Traceback\b)/.test(raw.trimStart()))return null;
    const analysis=raw.match(/^结构分析失败：([\s\S]*)$/);if(analysis&&analysis[0]===raw)return {key:'server.analysis_failed',parameters:{detail:analysis[1]}};
    const quote=value=>value.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
    for(const language of ['zh','en'])for(const [key,template] of Object.entries(i18n.messages[language])){
      if(!key.startsWith('server.')||key==='server.analysis_failed'||key==='server.invalid_json')continue;
      const names=[],tokens=template.split(/(\{\w+\})/g);let pattern='';
      for(const token of tokens){const match=token.match(/^\{(\w+)\}$/);if(!match){pattern+=quote(token);continue;}const name=match[1];names.push(name);
        const fields=key==='server.invalid_positive_integer'?'sample_num|steps':key==='server.invalid_boolean'?'geometric_exploration|single_step|random_step|random_step_size|parallel|save_repr_npz':key==='server.sample_field_missing'?'msa_path|fasta_path|init_pdb':null;
        pattern+=name==='exit_code'?'(-?\\d+)':['generated','expected'].includes(name)?'(\\d+)':name==='field'&&fields?`(${fields})`:'([^\\n]+)';
      }
      const match=raw.match(new RegExp(`^${pattern}$`));if(!match||match[0]!==raw)continue;
      const parameters={};names.forEach((name,index)=>{parameters[name]=['generated','expected','exit_code'].includes(name)?Number(match[index+1]):match[index+1];});return {key,parameters};
    }
    return null;
  }
  function messageParts(value){
    const rawValue=errorRaw(value),raw=typeof rawValue==='string'?rawValue:JSON.stringify(rawValue),legacy=legacyServerMessage(raw);
    const serverKey=value?.error_code?`server.${value.error_code}`:value?.code?`server.${value.code}`:null;
    let key=value?.key,parameters=value?.parameters||{};
    if(!key&&serverKey&&knownMessageKey(serverKey)){key=serverKey;parameters={...(legacy?.key===serverKey?legacy.parameters:{}),...(value.error_params&&typeof value.error_params==='object'?value.error_params:{})};}
    if(!key&&legacy){key=legacy.key;parameters=legacy.parameters;}
    if(!key||!knownMessageKey(key))return {summary:'',diagnostics:raw?[raw]:[]};
    const tokens=[...i18n.messages.en[key].matchAll(/\{(\w+)\}/g)].map(match=>match[1]);
    if(tokens.some(name=>parameters[name]===undefined||parameters[name]===null))return {summary:'',diagnostics:raw?[raw]:[]};
    const translatedParameters={},diagnostics=[];
    for(const [name,parameter] of Object.entries(parameters)){
      if(name==='detail'){translatedParameters[name]='';if(parameter!==undefined&&parameter!==null&&String(parameter))diagnostics.push(String(parameter));}
      else if(parameter&&typeof parameter==='object'&&(parameter.key||parameter.error_code||parameter.error!==undefined||parameter.message!==undefined)){const nested=messageParts(parameter);translatedParameters[name]=nested.summary;diagnostics.push(...nested.diagnostics);}
      else translatedParameters[name]=parameter;
    }
    const summary=t(key,translatedParameters).trim();
    if(serverKey&&raw&&(!legacy||legacy.key!==key)&&!diagnostics.includes(raw))diagnostics.push(raw);
    return {summary,diagnostics:[...new Set(diagnostics)]};
  }
  const messageText=value=>{const parts=messageParts(value);return parts.summary||parts.diagnostics.join('\n');};
  function renderMessage(node,value){
    const parts=messageParts(value);node.replaceChildren();
    if(parts.summary){const summary=document.createElement('span');summary.className='message-summary';summary.textContent=parts.summary;node.append(summary);}
    if(parts.diagnostics.length){const label=document.createElement('span');label.className='diagnostic-label';label.textContent=t('error.originalDiagnostic');node.append(label);const original=document.createElement('span');original.className='raw-diagnostic';original.dataset.rawDiagnostic='';original.textContent=parts.diagnostics.join('\n\n');node.append(original);}
  }
  const palette = ['#3977dd','#48a99a','#9b84d4','#e9a364','#65afda','#d885a5','#89b65e','#6b8fbd','#c5a74f','#7cb5b3','#b18868','#a299c8','#d59073','#73a78e','#7c9be2','#bf87b5','#91afcc','#d6b77a','#71bfc3','#b0aa6b'];
  const color = id => palette[(Number(id) - 1 + palette.length) % palette.length];
  const statusLabel=status=>t(`status.${status}`);
  const stages=[1,2,3,4,5];
  const defaults = {sample_num:200,geometric_exploration:true,single_step:false,steps:7,random_step:true,random_step_size:true,models:['Xray','NMR'],parallel:false,save_repr_npz:false,seed:null};
  const state = {config:{defaults},targets:[],filter:'all',query:'',drafts:[],id:null,target:null,analysis:null,viewer:null,models:new Map(),mode:'all',frame:0,clusterFocus:null,colorMode:'cluster',playing:false,playTimer:null,fps:3,clusterK:10,requestedClusterK:10,clusterTimer:null,analysisGeneration:0,detailLoaded:false,loadingAnalysis:false,pollBusy:false,fetchingViewer:false,connectionFailed:false,viewerGeneration:0};
  const iconProtein = '<svg class="protein-conformations" viewBox="0 0 24 24" aria-hidden="true"><path class="ribbon-variant" d="M3.9 11.2C2.2 6.1 7.1 2.4 10.4 5.2S10.5 12.1 13.7 14.3 21.6 15.1 21.3 19.8"/><path class="ribbon-variant ribbon-variant-far" d="M3 13.6C1.8 8.2 6.4 4.1 9.8 7.1S9.3 14.1 12.7 16.5 19.4 17.5 18.8 21.2"/><path class="ribbon-loop" d="M7 3.2C4.5 3.2 3.6 5.5 5.2 7l3.6 3.5c2.3 2.3-.3 4.3-2.4 2.6l-1.6-1.3C2.7 10 3.7 8.1 6 8.3l2.8.3c3.1.3 3.4 3.7.8 5.8L6.8 17c-1.7 1.6-1.3 3.6 1.1 3.8"/><path class="ribbon-return" d="M8 20.8c3.8.8 6.6-.7 7.8-4.1"/><path class="ribbon-beta" d="m11 17.8 5.2-8.5-1.6-.5 6.1-4.5v7.5l-1.8-1L13.7 19.3Z"/></svg>';
  const iconClock = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8"/><path d="M12 7v5l3 2"/></svg>';
  const iconArrow = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v12m-4-4 4 4 4-4M5 17v4h14v-4"/></svg>';
  const iconDelete = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 10v7m4-7v7"/></svg>';
  const activeStatuses = ['queued','running','analyzing'];
  const wait = ms => new Promise(resolve => setTimeout(resolve, ms));

  async function api(path, options = {}) {
    if(options.body&&new Blob([options.body]).size>64*1024*1024)throw new UIError('error.payloadSize');
    let response;try{response=await fetch(path, {...options,headers:{'Content-Type':'application/json',...(options.headers || {})}});}catch(cause){const error=new UIError('error.network',{detail:cause?.message||String(cause)});error.cause=cause;throw error;}
    const contentType = response.headers.get('content-type') || '';
    const data = contentType.includes('json') ? await response.json() : await response.text();
    if (!response.ok) {const error=data&&typeof data==='object'&&(data.error||data.detail)?new Error(typeof(data.error||data.detail)==='string'?(data.error||data.detail):JSON.stringify(data.error||data.detail)):typeof data==='string'&&data?new Error(data):new UIError('error.request',{status:response.status});error.status=response.status;error.code=data?.code;error.error_code=data?.error_code;error.error_params=data?.error_params;throw error;}
    return data;
  }
  function toast(value, error = false, undoId=null) {
    const item=document.createElement('div');item.className=`toast${error?' error':''}`;item._message=value;
    const text=document.createElement('span');text.className='toast-text';renderMessage(text,value);item.append(text);
    if(undoId){const button=document.createElement('button');button.type='button';button.className='toast-undo';button.dataset.i18n='delete.undo';button.textContent=t('delete.undo');button.onclick=async()=>{button.disabled=true;try{await api(`/api/recycle-bin/${encodeURIComponent(undoId)}/restore`,{method:'POST',body:'{}'});item.remove();toast(message('delete.restored'));await refreshTargets();}catch(restoreError){toast(restoreError,true);button.disabled=false;}};item.append(button);}
    $('#toasts').append(item);setTimeout(()=>item.remove(),undoId?15000:error?9000:5500);
  }
  function setError(node,value){node._message=value;renderMessage(node,value);node.hidden=!value;}
  function formatDate(value) {
    if (!value) return '—'; const date = new Date(typeof value === 'number' ? value * 1000 : value);
    if (Number.isNaN(date.getTime())) return String(value).replace('T',' ').slice(0,16);
    return new Intl.DateTimeFormat(i18n.language==='zh'?'zh-CN':'en-GB',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}).format(date);
  }
  function formatDuration(target) {
    const elapsed=Number(target.elapsed_seconds);
    const hasElapsed=target.status==='completed'&&target.elapsed_seconds!=null&&Number.isFinite(elapsed)&&elapsed>=0;
    if (!target.started_at&&!hasElapsed) return target.status==='completed' ? (target.existing?t('common.existing'):'—') : t('common.notStarted');
    const toMillis = value => typeof value === 'number' ? value * 1000 : new Date(value).getTime();
    const seconds = hasElapsed?elapsed:Math.max(0,Math.floor(((target.finished_at ? toMillis(target.finished_at) : Date.now()) - toMillis(target.started_at)) / 1000));
    if (!Number.isFinite(seconds)) return '—';
    if (seconds >= 3600) return t('common.hours',{hours:Math.floor(seconds/3600),minutes:Math.floor(seconds/60)%60});
    if (seconds >= 60) return t('common.minutes',{minutes:Math.floor(seconds/60),seconds:Math.floor(seconds)%60});
    return t('common.seconds',{seconds:Math.round(seconds*100)/100});
  }
  function badge(target) { return `<span class="status-badge status-${escapeHTML(target.status)}">${escapeHTML(statusLabel(target.status))}</span>`; }
  function sampleCount(target) { return target.sample_num ?? target.options?.sample_num ?? state.config.defaults.sample_num ?? 200; }
  function generatedCount(target) { return Number(target.generated ?? target.generated_count ?? 0); }
  function targetStage(target) { if(target.status==='completed')return t('stage.complete');if(target.status==='queued')return target.queue_position?t('queue.position',{position:target.queue_position}):t('queue.unassigned');const stage=Number(target.stage)||0;const text=stages.includes(stage)?t(`stage.${stage}`):t('queue.waitRuntime');return stage===3?`${text} · ${generatedCount(target)}/${sampleCount(target)}`:text; }
  function progress(target) { if(target.status==='completed')return 100; if(Number(target.stage)===3&&sampleCount(target))return Math.min(100,generatedCount(target)/sampleCount(target)*100);return ['running','analyzing'].includes(target.status)?null:0; }

  async function loadConfig() {
    try {
      const config = await api('/api/config'); state.config = {...config,defaults:{...defaults,...config.defaults}};
      state.device=typeof config.device==='string'?config.device:config.device?.name||config.device?.gpu_name||'';state.deviceConnected=true;updateDeviceLabel();
    } catch(error) {state.deviceConnected=false;updateDeviceLabel();}
  }
  function updateDeviceLabel(){const element=$('#device-label');delete element.dataset.i18n;const key=!state.deviceConnected?'device.disconnected':/NVIDIA|GPU/i.test(state.device||'')?'device.gpu':'device.local';element.textContent=t(key);element.title=state.deviceConnected&&state.device!=='local'&&!/[\u4e00-\u9fff]/.test(state.device||'')?state.device:t(key);}
  async function refreshTargets() {
    const response = await api('/api/targets'); state.targets = Array.isArray(response) ? response : response.targets || [];
    state.connectionFailed=false; $('.connection-error')?.remove(); renderHome();
    if(state.id) {
      const id=state.id;
      try { const detail=await api(`/api/targets/${encodeURIComponent(id)}`);if(id===state.id)updateDetailTarget(detail); } catch(error) { if(id===state.id&&!state.detailLoaded)renderDetailMissing(error); }
    }
  }
  async function poll() {
    if(state.pollBusy)return; state.pollBusy=true;
    try { await refreshTargets(); } catch(error) {
      if(!state.connectionFailed){state.connectionFailed=true;const bar=document.createElement('div');bar.className='connection-error';bar.innerHTML=`${label('connection.error')}<button type="button" data-i18n="connection.retry">${translated('connection.retry')}</button>`;bar.querySelector('button').onclick=()=>poll();$('.app-header').after(bar);}
      if(!state.targets.length)$('#target-list').innerHTML=`<div class="empty-state"><div class="empty-symbol">⌁</div><h3>${label('connection.waitTitle')}</h3><p>${label('connection.waitDescription')}</p></div>`;
    } finally { state.pollBusy=false; }
  }
  function renderHome() {
    const counts={all:state.targets.length,active:state.targets.filter(t=>activeStatuses.includes(t.status)).length,completed:state.targets.filter(t=>t.status==='completed').length,failed:state.targets.filter(t=>['failed','cancelled'].includes(t.status)).length};
    $('#target-total').textContent=counts.all; Object.entries(counts).forEach(([key,value])=>{$(`#count-${key}`).textContent=value;});
    const running=state.targets.find(t=>['running','analyzing'].includes(t.status)),queued=state.targets.filter(t=>t.status==='queued');
    $('#queue-summary').innerHTML=`${iconClock}<span>${running?`<strong>${escapeHTML(running.name)}</strong> · ${translated('queue.running',{stage:targetStage(running)})}`:`<strong>${translated('queue.idle')}</strong> · ${translated('queue.ready')}`}${queued.length?` <span class="dot-separator">·</span> ${translated('queue.waiting',{count:queued.length})}`:''}</span><span class="summary-end">${translated('queue.serial')}</span>`;
    const targets=state.targets.filter(target=>{
      const match=state.filter==='all'||(state.filter==='active'&&activeStatuses.includes(target.status))||(state.filter==='completed'&&target.status==='completed')||(state.filter==='failed'&&['failed','cancelled'].includes(target.status));
      return match&&String(target.name).toLowerCase().includes(state.query.toLowerCase());
    });
    if(!targets.length){$('#target-list').innerHTML=`<div class="empty-state"><div class="empty-symbol">⌁</div><h3>${translated(counts.all?'home.noMatchTitle':'home.emptyTitle')}</h3><p>${translated(counts.all?'home.noMatchDescription':'home.emptyDescription')}</p>${counts.all?'':`<button class="button secondary" data-action="new-target">${translated('hero.new')} →</button>`}</div>`;return;}
    $('#target-list').innerHTML=`<div class="table-head" aria-hidden="true"><span>${translated('home.targetColumn')}</span><span>${translated('home.countColumn')}</span><span>${translated('home.statusColumn')}</span><span class="time-head">${translated('home.dateColumn')}</span><span></span></div>${targets.map(target=>`<div class="target-row" data-target="${escapeHTML(target.id)}" role="button" tabindex="0" aria-label="${translated('home.openTarget',{name:target.name,status:statusLabel(target.status)})}"><div class="target-identity"><span class="target-icon">${iconProtein}</span><div style="min-width:0"><div class="target-name">${escapeHTML(target.name)}</div><div class="target-subtitle">${target.length?translated('common.residues',{count:Number(target.length)}):translated('common.pendingSequence')} <span>·</span> ${escapeHTML((target.options?.models || ['Xray','NMR']).join(' + '))}</div></div></div><div class="row-number">${Number(sampleCount(target))}</div><div class="stage-cell"><div class="stage-title">${badge(target)}<span>${escapeHTML(targetStage(target))}</span></div><div class="progress-track">${progress(target)===null?'<div class="progress-indeterminate"></div>':`<div class="progress-fill ${target.status==='completed'?'completed':''}" style="width:${progress(target)}%"></div>`}</div></div><div class="time-cell">${formatDate(target.created_at)}</div><div class="row-actions"><button type="button" class="row-delete icon-button" data-delete-target="${escapeHTML(target.id)}" title="${translated('delete.action')}" aria-label="${translated('delete.targetAria',{name:target.name})}">${iconDelete}</button><span class="row-chevron" aria-hidden="true">›</span></div></div>`).join('')}`;
  }
  function openDialog(id) { const dialog=$(`#${id}`); if(!dialog.open)dialog.showModal(); }
  function closeDialogs() { $$('dialog[open]').forEach(dialog=>dialog.close()); }
  function newDraft(overrides={}) { return {id:Math.random().toString(36).slice(2),name:'',msa_text:'',msa_name:'',fasta_text:'',init_pdb_text:'',msa_mode:'upload',options:{...state.config.defaults,models:[...(state.config.defaults.models || ['Xray','NMR'])]},...overrides}; }
  function openSubmit(draft) { state.drafts=[draft || newDraft()];renderDrafts();setError($('#submit-error'),'');openDialog('submit-dialog'); }
  const draftGenerationMode=options=>options.single_step?'single':options.random_step?'random':'fixed';
  function generationModeFields(draft){const mode=draftGenerationMode(draft.options);return `<fieldset class="generation-modes full"><legend class="subsection-label">${label('submit.generationMode')}</legend><div class="generation-mode-options">${[['single','single_step','submit.singleStep'],['random','random_step','submit.randomSteps'],['fixed',null,'submit.fixedSteps']].map(([value,option,key])=>`<label class="generation-mode check-field ${mode===value?'active':''}"><input type="radio" name="generation-${draft.id}" data-generation-mode="${value}" ${option?`data-option="${option}"`:''} ${mode===value?'checked':''}>${label(key)}</label>`).join('')}</div></fieldset>`;}
  function syncDraftGenerationMode(card,draft){const mode=draftGenerationMode(draft.options);if(mode==='single')draft.options.random_step=false;
    $$('[data-generation-mode]',card).forEach(input=>{input.checked=input.dataset.generationMode===mode;input.closest('.generation-mode').classList.toggle('active',input.checked);});
    const steps=$('[data-option="steps"]',card);steps.disabled=mode!=='fixed';steps.setAttribute('aria-disabled',String(steps.disabled));
  }
  function renderDrafts() {
    $('#draft-count').textContent=t('common.targets',{count:state.drafts.length});
    $('#draft-targets').innerHTML=state.drafts.map((draft,index)=>{
      const opts=draft.options;
      return `<section class="draft-card" data-draft="${draft.id}"><div class="draft-top"><span class="draft-index"><span data-i18n="submit.targetIndex" data-i18n-params='{"index":"${String(index+1).padStart(2,'0')}"}'>${translated('submit.targetIndex',{index:String(index+1).padStart(2,'0')})}</span></span>${state.drafts.length>1?`<button type="button" class="icon-button remove-draft" data-id="${draft.id}" data-i18n-title="submit.remove" data-i18n-aria-label="submit.removeAria" data-i18n-params='{"index":${index+1}}' title="${translated('submit.remove')}" aria-label="${translated('submit.removeAria',{index:index+1})}" style="width:24px;height:24px;font-size:16px">×</button>`:`<span class="field-hint" style="margin:0">${label('submit.independent')}</span>`}</div><div class="draft-body"><div class="field-space"><label class="field-label" for="name-${draft.id}">${label('submit.name')}</label><input class="text-field" id="name-${draft.id}" data-field="name" value="${escapeHTML(draft.name)}" data-i18n-placeholder="submit.namePlaceholder" placeholder="${translated('submit.namePlaceholder')}" autocomplete="off"></div><div class="field-label">${label('submit.msa')} <span style="color:#9aaac0;font-weight:400">A3M</span></div><div class="input-mode-tabs"><button type="button" data-msa-mode="upload" class="${draft.msa_mode==='upload'?'active':''}">${label('submit.upload')}</button><button type="button" data-msa-mode="paste" class="${draft.msa_mode==='paste'?'active':''}">${label('submit.paste')}</button></div><div data-upload-section ${draft.msa_mode==='paste'?'hidden':''}><label class="file-drop ${draft.msa_text?'loaded':''}" for="msa-${draft.id}"><span class="upload-icon">${draft.msa_text?'✓':'↥'}</span><strong>${translated(draft.msa_text?'submit.msaLoaded':'submit.dropMSA')}</strong><span class="file-name">${escapeHTML(draft.msa_name || t('submit.chooseFile'))}</span><input type="file" id="msa-${draft.id}" data-file="msa" accept=".a3m,.txt,.fasta,.fa" hidden></label></div><div data-paste-section ${draft.msa_mode!=='paste'?'hidden':''}><textarea class="msa-paste" data-field="msa_text" rows="6" spellcheck="false" placeholder=">query&#10;MSEQUENCE...&#10;>homolog_01&#10;MSEQUENCE...">${escapeHTML(draft.msa_text)}</textarea></div><div class="sequence-summary" data-sequence-summary>${sequenceSummary(draft.msa_text)}</div><div class="conformation-field"><div><label for="sample-${draft.id}">${label('submit.count')}</label><p>${label('submit.countHelp')}</p></div><input type="number" id="sample-${draft.id}" data-option="sample_num" value="${Number(opts.sample_num)}" min="1" max="10000" step="1" required></div><details class="advanced-options"><summary>${label('submit.advanced')} <span>${label('submit.defaults')}</span></summary><div class="advanced-grid"><div class="full"><div class="subsection-label">${label('submit.models')}</div><div class="model-checks">${['Xray','NMR'].map(model=>`<label class="check-field"><input type="checkbox" data-model="${model}" ${opts.models.includes(model)?'checked':''}>${model}</label>`).join('')}</div></div>${generationModeFields(draft)}<label class="check-field"><input type="checkbox" data-option="geometric_exploration" ${opts.geometric_exploration?'checked':''}>${label('submit.geometry')}</label><label class="check-field"><input type="checkbox" data-option="random_step_size" ${opts.random_step_size?'checked':''}>${label('submit.randomStepSize')}</label><div><label class="field-label" for="steps-${draft.id}">${label('submit.stepCount')}</label><input class="number-field" type="number" id="steps-${draft.id}" data-option="steps" value="${Number(opts.steps)}" min="1" max="1000"><p class="field-hint">${label('submit.stepsHelp')}</p></div><div><label class="field-label" for="seed-${draft.id}">${label('submit.seed')}</label><input class="text-field" type="number" id="seed-${draft.id}" data-option="seed" value="${opts.seed??''}" data-i18n-placeholder="submit.seedPlaceholder" placeholder="${translated('submit.seedPlaceholder')}" min="0"></div><label class="check-field full"><input type="checkbox" data-option="save_repr_npz" ${opts.save_repr_npz?'checked':''}>${label('submit.saveRepresentation')}</label><div class="full"><label class="field-label" for="fasta-${draft.id}">${label('submit.fasta')}</label><div class="file-control"><label class="button secondary" for="fasta-${draft.id}">${label('submit.chooseFASTA')}</label><span data-file-label="fasta">${escapeHTML(draft.fasta_name||t('submit.noFile'))}</span><input type="file" id="fasta-${draft.id}" data-file="fasta" accept=".fa,.fasta,.faa,.txt" hidden></div><p class="field-hint">${label('submit.fastaHelp')}</p></div><div class="full"><label class="field-label" for="pdb-${draft.id}">${label('submit.initialPDB')}</label><div class="file-control"><label class="button secondary" for="pdb-${draft.id}">${label('submit.choosePDB')}</label><span data-file-label="pdb">${escapeHTML(draft.pdb_name||t('submit.noFile'))}</span><input type="file" id="pdb-${draft.id}" data-file="pdb" accept=".pdb" hidden></div><p class="field-hint">${label('submit.pdbHelp')}${draft.init_pdb_text?label('submit.pdbLoaded'):''}</p></div></div></details></div></section>`;
    }).join('');
    $$('.draft-card').forEach(card=>{ $('[data-option="sample_num"]',card).max=state.config.max_conformations||2000;$('[data-option="steps"]',card).max=100;$('[data-option="seed"]',card).max=4294967295;$('[data-field="name"]',card).maxLength=120;bindDraft(card); });
  }
  function parseMSA(text) {
    const rows=String(text||'').trim().split(/\r?\n/);let count=0,query='',inFirst=false;
    for(const row of rows){if(row.startsWith('>')){count++;inFirst=count===1;}else if(inFirst&&row.trim()&&!row.startsWith('#'))query+=row.trim();}
    const first=rows.find(row=>row.trim());
    const valid=Boolean(first?.startsWith('>')&&first.slice(1).trim()&&query.length&&/^[ACDEFGHIKLMNPQRSTVWYX]+$/.test(query));
    return {count,length:query.length,query,valid};
  }
  function sequenceSummary(text) { if(!text.trim())return label('submit.firstQuery');const parsed=parseMSA(text);if(!parsed.count||!parsed.length)return label('submit.invalidMSA');if(!parsed.valid)return label('submit.invalidQuery');return `<span>✓ ${translated('common.residues',{count:parsed.length})}</span><span>· ${translated('submit.alignmentCount',{count:parsed.count.toLocaleString(i18n.language==='zh'?'zh-CN':'en')})}</span>`; }
  function updateDraftSequence(card,draft){const summary=$('[data-sequence-summary]',card);summary.innerHTML=sequenceSummary(draft.msa_text);const parsed=parseMSA(draft.msa_text);summary.classList.toggle('valid',parsed.valid);summary.classList.toggle('error',Boolean(draft.msa_text.trim()&&!parsed.valid));}
  function bindDraft(card) {
    const draft=state.drafts.find(item=>item.id===card.dataset.draft);
    $$('[data-field]',card).forEach(input=>input.addEventListener('input',()=>{draft[input.dataset.field]=input.value;if(input.dataset.field==='msa_text')updateDraftSequence(card,draft);}));
    $$('[data-option]',card).filter(input=>!input.dataset.generationMode).forEach(input=>input.addEventListener('input',()=>{if(input.disabled)return;draft.options[input.dataset.option]=input.type==='checkbox'?input.checked:input.dataset.option==='seed'&&input.value===''?null:Number(input.value);}));
    $$('[data-generation-mode]',card).forEach(input=>input.addEventListener('change',()=>{if(!input.checked)return;draft.options.single_step=input.dataset.generationMode==='single';draft.options.random_step=input.dataset.generationMode==='random';syncDraftGenerationMode(card,draft);}));
    $$('[data-model]',card).forEach(input=>input.addEventListener('change',()=>{draft.options.models=$$('[data-model]:checked',card).map(node=>node.dataset.model);}));
    $$('[data-msa-mode]',card).forEach(button=>button.addEventListener('click',()=>{draft.msa_mode=button.dataset.msaMode;$$('[data-msa-mode]',card).forEach(node=>node.classList.toggle('active',node===button));$('[data-upload-section]',card).hidden=draft.msa_mode!=='upload';$('[data-paste-section]',card).hidden=draft.msa_mode!=='paste';$('[data-field="msa_text"]',card).value=draft.msa_text;}));
    $$('[data-file]',card).forEach(input=>input.addEventListener('change',async()=>{const file=input.files[0];if(file)await readDraftFile(card,draft,input.dataset.file,file);}));
    const drop=$('.file-drop',card);['dragenter','dragover'].forEach(event=>drop.addEventListener(event,e=>{e.preventDefault();drop.classList.add('dragover');}));['dragleave','drop'].forEach(event=>drop.addEventListener(event,e=>{e.preventDefault();drop.classList.remove('dragover');}));drop.addEventListener('drop',async event=>{if(event.dataTransfer.files[0])await readDraftFile(card,draft,'msa',event.dataTransfer.files[0]);});
    updateDraftSequence(card,draft);
    syncDraftGenerationMode(card,draft);
  }
  async function readDraftFile(card,draft,type,file) {
    try {if(file.size>64*1024*1024)throw new UIError('error.fileSize');const text=await file.text();if(type==='msa'){draft.msa_text=text;draft.msa_name=file.name;if(!draft.name){draft.name=file.name.replace(/\.[^.]+$/,'');$('[data-field="name"]',card).value=draft.name;}const drop=$('.file-drop',card);drop.classList.add('loaded');$('.upload-icon',drop).textContent='✓';$('strong',drop).textContent=t('submit.msaLoaded');$('.file-name',drop).textContent=file.name;$('[data-field="msa_text"]',card).value=text;updateDraftSequence(card,draft);}else if(type==='fasta'){draft.fasta_text=text;draft.fasta_name=file.name;$('[data-file-label="fasta"]',card).textContent=file.name;}else{draft.init_pdb_text=text;draft.pdb_name=file.name;$('[data-file-label="pdb"]',card).textContent=file.name;}}catch(error){toast(error,true);}
  }
  async function loadExample() {
    const button=$('#example-button');button.disabled=true;
    try {const example=await api('/api/example');openSubmit(newDraft({name:example.name||'example',msa_text:example.msa_text||'',msa_name:example.msa_filename||`${example.name||'example'}.a3m`,fasta_text:example.fasta_text||'',init_pdb_text:example.init_pdb_text||''}));toast(message('notice.exampleLoaded'));}catch(error){toast(error,true);}finally{button.disabled=false;}
  }
  async function submitDrafts(event) {
    event.preventDefault();const button=$('#submit-button');if(button.disabled)return;setError($('#submit-error'),'');
    try {
      const targets=state.drafts.map((draft,index)=>{
        if(!draft.name.trim())throw new UIError('error.targetName',{index:index+1});
        const parsed=parseMSA(draft.msa_text);if(!parsed.valid)throw new UIError('error.targetMSA',{index:index+1});
        const maximum=state.config.max_conformations||2000;
        if(!Number.isInteger(draft.options.sample_num)||draft.options.sample_num<1||draft.options.sample_num>maximum)throw new UIError('error.targetCount',{index:index+1,maximum});
        if(!draft.options.models.length)throw new UIError('error.targetModels',{index:index+1});
        const mode=draftGenerationMode(draft.options);if(mode==='fixed'&&(!Number.isInteger(draft.options.steps)||draft.options.steps<1||draft.options.steps>100))throw new UIError('error.targetSteps',{index:index+1});
        if(draft.options.seed!==null&&(!Number.isInteger(draft.options.seed)||draft.options.seed<0||draft.options.seed>=2**32))throw new UIError('error.targetSeed',{index:index+1});
        const options={...draft.options,models:[...draft.options.models],single_step:mode==='single',random_step:mode==='random',steps:mode==='single'?1:mode==='random'?7:draft.options.steps};
        return {name:draft.name.trim(),msa_text:draft.msa_text,fasta_text:draft.fasta_text||undefined,init_pdb_text:draft.init_pdb_text||undefined,options};
      });
      button.disabled=true;button.textContent=t('common.submitting');const response=await api('/api/targets',{method:'POST',body:JSON.stringify({targets})});closeDialogs();toast(message('notice.queued',{count:response.targets?.length||targets.length}));await refreshTargets();if((response.targets||[]).length===1)location.hash=`target/${response.targets[0].id}`;
    }catch(error){setError($('#submit-error'),error);}finally{button.disabled=false;button.innerHTML=`${label('submit.button')} <span>→</span>`;}
  }
  async function submitJSON(event) {
    event.preventDefault();const button=$('#import-submit');setError($('#import-error'),'');
    try {let config;try{config=JSON.parse($('#json-text').value);}catch(error){throw new UIError('error.jsonSyntax');}if(!config||typeof config!=='object'||!Array.isArray(config.samples))throw new UIError('error.jsonObject');const files=await Promise.all(Array.from($('#json-msa-files').files).map(async file=>({name:file.name,content:await file.text()})));button.disabled=true;button.textContent=t('common.submitting');const response=await api('/api/import',{method:'POST',body:JSON.stringify({config,files})});closeDialogs();toast(message('notice.jsonQueued',{count:response.targets?.length||0}));await refreshTargets();if(response.targets?.length===1)location.hash=`target/${response.targets[0].id}`;}catch(error){setError($('#import-error'),error);}finally{button.disabled=false;button.innerHTML=`${label('import.button')} <span>→</span>`;}
  }
  function stopPlayback(){if(state.playTimer)clearInterval(state.playTimer);state.playTimer=null;state.playing=false;const button=$('#play-button');if(button){button.textContent='▶';button.setAttribute('aria-label',t('viewer.play'));}}
  function destroyViewer(){stopPlayback();state.viewerGeneration++;state.fetchingViewer=false;state.viewerError=null;if(state.viewer){try{state.viewer.clear();}catch(error){/* A previously removed canvas may already be gone. */}}state.viewer=null;state.models=new Map();}
  function route(){const match=location.hash.match(/^#target\/(.+)$/);const id=match?decodeURIComponent(match[1]):null;if(id===state.id&&state.detailLoaded)return;clearTimeout(state.clusterTimer);state.clusterTimer=null;state.analysisGeneration++;destroyViewer();state.id=id;state.target=null;state.analysis=null;state.detailLoaded=false;state.loadingAnalysis=false;state.analysisFailed=false;state.clusterK=10;state.requestedClusterK=10;state.frame=0;state.mode='all';state.clusterFocus=null;state.colorMode='cluster';$('#home-view').hidden=Boolean(id);$('#detail-view').hidden=!id;if(id){$('#detail-view').innerHTML=`<div class="empty-state"><span class="loading-spinner"></span><h3>${label('detail.loading')}</h3></div>`;const target=state.targets.find(item=>item.id===id);if(target)updateDetailTarget(target);else poll();}window.scrollTo({top:0,behavior:'instant'});}
  function renderDetailMissing(value){state.detailMissingError=value;$('#detail-view').innerHTML=`<div class="breadcrumb"><a href="#">${label('nav.workspace')}</a><span>›</span>${label('detail.title')}</div><div class="empty-state panel"><h3>${label('detail.unavailable')}</h3><div class="missing-error" id="detail-missing-error"></div><a class="button secondary" href="#">${label('detail.back')}</a></div>`;renderMessage($('#detail-missing-error'),value);}
  function updateDetailTarget(target){if(!state.id||target.id!==state.id)return;const previous=state.target;state.target=target;
    if(!state.detailLoaded){state.detailMissingError=null;renderDetailShell(target);state.detailLoaded=true;}
    updateDetailMeta(target);
    const completed=target.status==='completed';
    if(completed&&!state.analysis&&!state.loadingAnalysis&&!state.analysisFailed)loadAnalysis();
    if(previous&&previous.status!==target.status&&!completed&&state.analysis){state.analysis=null;destroyViewer();$('#result-area').hidden=true;$('#running-area').hidden=false;}
  }
  function renderDetailShell(target){$('#detail-view').innerHTML=`<div class="breadcrumb"><a href="#">${label('nav.workspace')}</a><span>›</span><span>${escapeHTML(target.name)}</span></div><div class="detail-heading"><div><div class="eyebrow" style="margin-bottom:6px">${label('detail.eyebrow')}</div><h1>${escapeHTML(target.name)}</h1><div class="detail-subtitle" id="detail-subtitle"></div></div><div class="detail-actions" id="detail-actions"></div></div><div class="stage-strip" id="stage-strip"></div><div id="detail-error" class="error-message" hidden role="alert"></div><div id="running-area"><div class="running-layout"><div class="panel running-panel"><div class="running-illustration">${iconProtein}</div><div><h2 class="running-title" id="running-title">${label('detail.preparing')}</h2><p class="running-description" id="running-description"></p><div class="running-count" id="running-count"></div></div></div><div class="panel"><div class="facts-grid" id="run-facts"></div></div></div></div><div id="result-area" hidden></div><details class="panel log-panel" id="detail-log"><summary><span>${label('detail.logs')}</span><a class="log-link" href="/api/targets/${encodeURIComponent(target.id)}/log" target="_blank" rel="noopener">${label('detail.fullLog')} ↗</a></summary><pre class="log-output" id="log-output">${translated('detail.waitLog')}</pre></details><footer class="page-footer"><span>trFlow · ${label('app.brandCaption')}</span><span>${label('detail.footer')}</span></footer>`;}
  function updateDetailMeta(target){const length=target.length||state.analysis?.length;$('#detail-subtitle').innerHTML=`${badge(target)}<span>${length?translated('common.residues',{count:Number(length)}):translated('common.pendingSequence')}</span><span>${translated('common.conformations',{count:Number(sampleCount(target))})}</span><span>${escapeHTML((target.options?.models||[]).join(' + '))}</span><span>${translated('detail.submitted',{date:formatDate(target.created_at)})}</span>`;
    const action=target.status==='completed'?`<a class="button secondary" data-detail-action="download" role="link">${iconArrow}${label('detail.download')}</a>`:activeStatuses.includes(target.status)?`<button class="button danger-button" data-detail-action="cancel">${label('detail.cancel')}</button>`:`<button class="button secondary" data-detail-action="retry">${label('detail.retry')}</button>`;
    $('#detail-actions').innerHTML=`${action}<button class="button danger-button" data-detail-action="delete" title="${translated('delete.action')}">${iconDelete}${label('delete.action')}</button>`;
    updateDownloadAction();
    const stage=Number(target.stage||0),completed=target.status==='completed';
    $('#stage-strip').innerHTML=stages.map(number=>{const done=completed||stage>number,current=!completed&&stage===number,skipped=number===2&&target.options?.geometric_exploration===false;return `<div class="stage-item ${done?'done':''} ${current?'current':''}"><span class="stage-step">${skipped&&done?'−':done?'✓':String(number).padStart(2,'0')}</span><div><strong>${translated(`stage.${number}`)}</strong><p>${translated(skipped?'stage.skipped':`stage.caption.${number}`)}</p></div></div>`;}).join('');
    const pairs={queued:[t('detail.queueTitle'),t('detail.queueDescription',{position:target.queue_position?`#${target.queue_position}`:t('detail.queueUnassigned')})],running:[targetStage(target),t('detail.runningDescription')],analyzing:[t('detail.analysisTitle'),t('detail.analysisDescription')],completed:[t('detail.completeTitle'),t('detail.completeDescription')],failed:[t('detail.failedTitle'),t('detail.failedDescription')],cancelled:[t('detail.cancelledTitle'),t('detail.cancelledDescription')]};
    if(!state.analysisFailed){const pair=pairs[target.status]||pairs.running;$('#running-title').textContent=pair[0];$('#running-description').textContent=pair[1];$('#running-count').textContent=target.status==='queued'?t('detail.serial'):generatedCount(target)?t('detail.generated',{generated:generatedCount(target),total:sampleCount(target)}):t('detail.elapsed',{time:formatDuration(target)});}else if(!state.analysis){$('#running-title').textContent=t('analysis.unavailable');renderMessage($('#running-description'),state.analysisError);$('#running-count').innerHTML=`<button class="button secondary" data-detail-action="analysis-retry">${label('analysis.retry')}</button>`;}
    $('#run-facts').innerHTML=`<div><div class="fact-label">${label('detail.count')}</div><div class="fact-value big">${Number(sampleCount(target))}</div></div><div><div class="fact-label">${label('detail.time')}</div><div class="fact-value">${formatDuration(target)}</div></div><div><div class="fact-label">${label('submit.geometry')}</div><div class="fact-value">${translated(target.options?.geometric_exploration===false?'common.disabled':'common.enabled')}</div></div><div><div class="fact-label">${label('submit.seed')}</div><div class="fact-value">${target.options?.seed??translated('common.random')}</div></div>`;
    const error=Boolean(target.error||target.error_code);if(error)renderMessage($('#detail-error'),target);else $('#detail-error').replaceChildren();$('#detail-error').hidden=!error;
    const log=$('#log-output');const nearBottom=log.scrollHeight-log.scrollTop-log.clientHeight<45;const text=(Array.isArray(target.log_tail)?target.log_tail.join('\n'):target.log_tail)||t((target.status==='completed'&&target.existing)?'detail.existingNoLog':'detail.waitLog');if(log.textContent!==text){log.textContent=text;if(nearBottom)log.scrollTop=log.scrollHeight;}
    if(!state.analysis){$('#running-area').hidden=false;$('#result-area').hidden=true;}
  }
  async function detailAction(action){const id=state.id;if(!id)return;const button=$(`[data-detail-action="${action}"]`);if(button)button.disabled=true;
    try{const response=await api(`/api/targets/${encodeURIComponent(id)}/${action}`,{method:'POST',body:'{}'});toast(message(action==='cancel'?'notice.cancelRequested':'notice.requeued'));await refreshTargets();if(action==='retry'&&response.targets?.[0])location.hash=`target/${response.targets[0].id}`;}catch(error){toast(error,true);}finally{if(button)button.disabled=false;}}
  function updateDownloadAction(){const link=$('[data-detail-action="download"]',$('#detail-actions'));if(!link)return;
    const ready=Boolean(state.analysis&&!state.loadingAnalysis&&state.requestedClusterK===state.analysis.k);
    link.setAttribute('aria-disabled',String(!ready));link.title=t(ready?'detail.downloadHint':'detail.downloadWait');link.style.opacity=ready?'':'.55';link.style.cursor=ready?'':'wait';
    if(ready){link.href=`/api/targets/${encodeURIComponent(state.id)}/download?k=${state.analysis.k}`;link.removeAttribute('tabindex');}else{link.removeAttribute('href');link.setAttribute('tabindex','-1');}
  }
  async function loadAnalysis(k=state.requestedClusterK){const id=state.id;if(!id)return;const generation=++state.analysisGeneration;state.requestedClusterK=k;state.loadingAnalysis=true;state.analysisFailed=false;updateDownloadAction();
    const panel=$('#cluster-panel');panel?.classList.add('busy-panel');
    try{const analysis=await api(`/api/targets/${encodeURIComponent(id)}/analysis?k=${encodeURIComponent(k)}`);if(id!==state.id||generation!==state.analysisGeneration)return;const first=!state.analysis;state.analysis=analysis;state.clusterK=analysis.k;state.requestedClusterK=analysis.k;state.clusterFocus=null;state.frame=Math.min(state.frame,Math.max(0,analysis.count-1));if(first)renderResults();else updateAnalysisViews();if(first)await loadViewer();else refreshViewer();}catch(error){if(id===state.id&&generation===state.analysisGeneration){state.analysisFailed=true;state.analysisError=error;toast(message('notice.analysisError',{error}),true);if(!state.analysis)updateDetailMeta(state.target);}}finally{if(id===state.id&&generation===state.analysisGeneration){state.loadingAnalysis=false;$('#cluster-panel')?.classList.remove('busy-panel');updateDownloadAction();}}
  }
  function renderResults(){const target=state.target;$('#running-area').hidden=true;$('#result-area').hidden=false;$('#result-area').innerHTML=`<div class="completion-stats" id="completion-stats"></div><div class="results-layout"><section class="panel viewer-panel"><div class="panel-header"><div><h2>${label('viewer.title')}</h2><p>${label('viewer.help')}</p></div><button class="icon-button" id="viewer-reset" data-i18n-title="viewer.reset" title="${translated('viewer.reset')}" data-i18n-aria-label="viewer.reset" aria-label="${translated('viewer.reset')}" style="width:28px;height:28px;font-size:14px">↺</button></div><div class="viewer-toolbar"><div class="segment-group" role="tablist" data-i18n-aria-label="viewer.modeAria" aria-label="${translated('viewer.modeAria')}"><button data-view-mode="all" role="tab" aria-selected="true" class="active">${label('viewer.all')}</button><button data-view-mode="representatives" role="tab" aria-selected="false">${label('viewer.representatives')}</button><button data-view-mode="single" role="tab" aria-selected="false">${label('viewer.single')}</button></div><label class="toolbar-end"><span class="sr-only">${label('viewer.colorAria')}</span><select id="color-mode"><option value="cluster" data-i18n="viewer.byCluster">${translated('viewer.byCluster')}</option><option value="confidence" data-i18n="viewer.byConfidence">${translated('viewer.byConfidence')}</option></select></label></div><div class="viewer-wrap"><div id="molecule-viewer" class="molecule-viewer" data-i18n-aria-label="viewer.aria" aria-label="${translated('viewer.aria')}"></div><div class="viewer-overlay" id="viewer-loading"><span class="loading-spinner"></span><span id="viewer-load-message">${label('viewer.loading')}</span></div><div class="viewer-corner">${label('viewer.aligned')}</div><div class="viewer-caption" id="viewer-caption"></div></div><div class="ensemble-legend" id="ensemble-legend"></div><div class="viewer-controls"><div class="playback-row"><button class="play-button" id="play-button" data-i18n-aria-label="viewer.play" aria-label="${translated('viewer.play')}" data-i18n-title="viewer.playTitle" title="${translated('viewer.playTitle')}">▶</button><input type="range" id="frame-slider" class="playback-slider" min="0" max="0" value="0" step="1" data-i18n-aria-label="viewer.current" aria-label="${translated('viewer.current')}"><span class="frame-label" id="frame-label">1 / 1</span><label><span class="sr-only">${label('viewer.speed')}</span><select id="playback-speed"><option value="1">1 fps</option><option value="3" selected>3 fps</option><option value="6">6 fps</option><option value="10">10 fps</option></select></label></div><div class="structure-select-row"><span>${label('viewer.select')}</span><select id="structure-select" data-i18n-aria-label="viewer.selectAria" aria-label="${translated('viewer.selectAria')}"></select></div></div><div class="selected-info" id="selected-info"></div><div class="viewer-footer"><span id="visible-count">${label('viewer.alignedReference')}</span><button id="viewer-image">${label('viewer.export')} ↗</button></div></section><section class="panel cluster-panel" id="cluster-panel"><div class="panel-header"><div><h2>${label('cluster.title')}</h2><p>${label('cluster.description')}</p></div><span class="count-badge" id="cluster-badge">${translated('common.clusters',{count:10})}</span></div><div class="cluster-controls"><div class="cluster-control-heading"><label for="cluster-number">${label('cluster.count')}</label><div class="cluster-value"><input type="number" id="cluster-number" min="1" max="10" value="10" data-i18n-aria-label="cluster.count" aria-label="${translated('cluster.count')}"><span style="font-size:10px;color:#96a7bf">${label('cluster.unit')}</span></div></div><input type="range" id="cluster-slider" class="cluster-slider" min="1" max="10" value="10" step="1" data-i18n-aria-label="cluster.adjustAria" aria-label="${translated('cluster.adjustAria')}"><div class="range-labels"><span>${translated('common.clusters',{count:1})}</span><span id="cluster-max-label">${translated('common.clusters',{count:10})}</span></div><p class="cluster-hint">${label('cluster.help')}</p></div><div class="chart-section"><div class="chart-heading"><span>${label('cluster.population')}</span><span>${label('cluster.clickFilter')}</span></div><div class="population-chart" id="population-chart"></div></div><div class="chart-section"><div class="chart-heading"><span>${label('cluster.embedding')}</span><span>${label('cluster.clickStructure')}</span></div><svg id="embedding-chart" class="embedding-svg" viewBox="0 0 310 200" role="img" data-i18n-aria-label="cluster.embeddingAria" aria-label="${translated('cluster.embeddingAria')}"></svg><p class="embedding-note" id="embedding-note"></p><div class="analysis-stats" id="analysis-stats"></div></div></section></div><div class="result-bottom-grid"><section class="panel"><div class="panel-header"><div><h2>${label('viewer.representatives')}</h2><p>${label('cluster.representativeHelp')}</p></div><span class="count-badge" id="representative-count"></span></div><div class="table-wrap"><table class="representative-table"><thead><tr><th>${label('cluster.column')}</th><th>${label('cluster.populationColumn')}</th><th>${label('cluster.representativeColumn')}</th><th>${label('cluster.rmsdColumn')}</th></tr></thead><tbody id="representative-rows"></tbody></table></div></section><section class="panel"><div class="panel-header"><div><h2>${label('distance.title')}</h2><p>${label('distance.description')}</p></div></div><div class="distance-wrap"><canvas class="distance-canvas" id="distance-heatmap" width="250" height="250" role="img" data-i18n-aria-label="distance.aria" aria-label="${translated('distance.aria')}"></canvas><div class="heatmap-legend"><span>0</span><span class="heatmap-gradient"></span><span id="heatmap-max"></span></div><div class="heatmap-tooltip" id="heatmap-tooltip">${label('distance.hover')}</div><p class="heatmap-caption">${label('distance.caption')}</p></div></section></div>`;
    const clusterMinimum=$('.range-labels > span:first-child',$('#cluster-panel'));
    clusterMinimum.dataset.i18n='common.clusters';clusterMinimum.dataset.i18nParams='{"count":1}';
    bindResultEvents();updateAnalysisViews();
  }
  function bindResultEvents(){
    $$('[data-view-mode]').forEach(button=>button.addEventListener('click',()=>{stopPlayback();state.mode=button.dataset.viewMode;state.clusterFocus=null;refreshViewer();updateViewerControls();}));
    $('#color-mode').addEventListener('change',event=>{state.colorMode=event.target.value;refreshViewer();});
    $('#viewer-reset').addEventListener('click',()=>{if(state.viewer){state.viewer.zoomTo();state.viewer.zoom(1.5);state.viewer.render();}});
    $('#viewer-image').addEventListener('click',()=>{if(!state.viewer)return;try{const link=document.createElement('a');link.href=state.viewer.pngURI();link.download=`${state.target.name}_ensemble.png`;link.click();}catch(error){toast(message('error.export'),true);}});
    $('#play-button').addEventListener('click',()=>{if(state.playing)stopPlayback();else startPlayback();});
    $('#frame-slider').addEventListener('input',event=>{stopPlayback();selectStructure(Number(event.target.value));});
    $('#structure-select').addEventListener('change',event=>{stopPlayback();selectStructure(Number(event.target.value));});
    $('#playback-speed').addEventListener('change',event=>{state.fps=Number(event.target.value);if(state.playing)startPlayback();});
    const clusterInput=event=>{const maximum=state.analysis.count;const requested=Math.max(1,Math.min(maximum,Math.trunc(Number(event.target.value))||1));$('#cluster-number').value=requested;$('#cluster-slider').value=requested;if(requested===state.requestedClusterK&&!state.analysisFailed)return;
      clearTimeout(state.clusterTimer);state.clusterTimer=null;state.analysisGeneration++;state.requestedClusterK=requested;state.analysisFailed=false;state.loadingAnalysis=requested!==state.clusterK;updateDownloadAction();
      if(!state.loadingAnalysis){$('#cluster-panel')?.classList.remove('busy-panel');return;}const id=state.id;state.clusterTimer=setTimeout(()=>{state.clusterTimer=null;if(id===state.id&&requested===state.requestedClusterK)loadAnalysis(requested);},350);
    };
    $('#cluster-number').addEventListener('input',clusterInput);$('#cluster-number').addEventListener('change',clusterInput);$('#cluster-slider').addEventListener('input',clusterInput);
    $('#distance-heatmap').addEventListener('mousemove',event=>{const canvas=event.target,box=canvas.getBoundingClientRect(),order=canvas._structureOrder;if(!order?.length)return;const x=Math.min(order.length-1,Math.max(0,Math.floor((event.clientX-box.left)/box.width*order.length))),y=Math.min(order.length-1,Math.max(0,Math.floor((event.clientY-box.top)/box.height*order.length)));const a=order[x],b=order[y],distance=state.analysis.distance_matrix[a]?.[b];$('#heatmap-tooltip').textContent=t('distance.pair',{a:a+1,b:b+1,rmsd:Number(distance||0).toFixed(2)});});
    $('#distance-heatmap').addEventListener('mouseleave',()=>{$('#heatmap-tooltip').textContent=t('distance.hover');});
  }
  function analysisStructure(index){return state.analysis?.structures.find(item=>item.index===Number(index));}
  function updateAnalysisViews(){const a=state.analysis;if(!a)return;$('#running-area').hidden=true;$('#result-area').hidden=false;
    updateDownloadAction();
    const length=state.target.length||a.length;if(length){const span=$('#detail-subtitle>span:nth-child(2)');if(span)span.textContent=t('common.residues',{count:length});}
    const avgConfidence=a.structures.filter(s=>s.mean_plddt!==null&&s.mean_plddt!==undefined).map(s=>Number(s.mean_plddt));const meanConfidence=avgConfidence.length?avgConfidence.reduce((x,y)=>x+y,0)/avgConfidence.length:null;
    $('#completion-stats').innerHTML=`<div class="completion-stat">${label('analysis.generated')} <strong>${translated('common.conformations',{count:a.count})}</strong></div><div class="completion-stat">${label('analysis.length')} <strong>${translated('common.residues',{count:a.length})}</strong></div><div class="completion-stat">${label('analysis.clusters')} <strong>${translated('analysis.states',{count:a.k})}</strong></div><div class="completion-stat">${label('analysis.time')} <strong>${formatDuration(state.target)}</strong></div>`;
    $('#cluster-badge').textContent=t('common.clusters',{count:a.k});$('#cluster-number').max=a.count;$('#cluster-number').value=state.requestedClusterK;$('#cluster-slider').max=a.count;$('#cluster-slider').value=state.requestedClusterK;$('#cluster-max-label').textContent=t('common.clusters',{count:a.count});
    $('#population-chart').innerHTML=a.clusters.map(cluster=>{const max=Math.max(...a.clusters.map(c=>c.count));return `<button class="population-column" data-cluster="${cluster.id}" aria-label="${translated('cluster.populationAria',{id:cluster.id,count:cluster.count})}"><span class="population-count">${cluster.count}</span><span class="population-bar" style="height:${Math.max(3,cluster.count/max*72)}px;background:${color(cluster.id)}"></span><span class="population-label">${cluster.id}</span></button>`;}).join('');
    $('#population-chart').classList.toggle('many-clusters',a.k>20);
    $$('[data-cluster]',$('#population-chart')).forEach(button=>button.addEventListener('click',()=>focusCluster(Number(button.dataset.cluster))));
    $('#ensemble-legend').innerHTML=a.clusters.map(cluster=>`<button class="legend-button" data-legend-cluster="${cluster.id}" aria-label="${translated('cluster.filterAria',{id:cluster.id})}"><span class="color-dot" style="background:${color(cluster.id)}"></span>${translated('common.clusterShort',{id:cluster.id})}<span style="color:#b0bccd">(${cluster.count})</span></button>`).join('');
    $$('[data-legend-cluster]').forEach(button=>button.addEventListener('click',()=>focusCluster(Number(button.dataset.legendCluster))));
    renderEmbedding();renderHeatmap();
    $('#representative-count').textContent=t('common.items',{count:a.k});$('#representative-rows').innerHTML=a.clusters.map(cluster=>{const structure=analysisStructure(cluster.representative);return `<tr data-structure="${cluster.representative}" tabindex="0" aria-label="${translated('cluster.representativeAria',{cluster:cluster.id,index:cluster.representative+1})}"><td><span class="cluster-cell"><span class="color-dot" style="background:${color(cluster.id)}"></span>${translated('common.clusterShort',{id:cluster.id})}</span></td><td>${cluster.count}</td><td>${escapeHTML(structureLabel(structure))}</td><td>${Number(cluster.mean_rmsd||0).toFixed(2)} Å</td></tr>`;}).join('');
    $$('[data-structure]',$('#representative-rows')).forEach(row=>{row.onclick=()=>{stopPlayback();selectStructure(Number(row.dataset.structure));};row.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();row.click();}};});
    $('#structure-select').innerHTML=a.structures.map(structure=>`<option value="${structure.index}">${escapeHTML(structureLabel(structure))} · ${translated('common.clusterShort',{id:structure.cluster})}${structure.model?' · '+escapeHTML(structure.model):''}</option>`).join('');$('#frame-slider').max=a.count-1;
    const rmsds=a.structures.map(s=>Number(s.rmsd_to_reference||0));$('#analysis-stats').innerHTML=`<div><strong>${a.count}</strong><span>${label('analysis.total')}</span></div><div><strong>${Math.max(...rmsds).toFixed(1)} Å</strong><span>${label('analysis.offset')}</span></div><div><strong>${meanConfidence!==null?(meanConfidence*100).toFixed(0):'—'}</strong><span>${label('analysis.plddt')}</span></div>`;
    updateViewerControls();
  }
  function structureLabel(structure){if(!structure)return '—';return `${t('viewer.structure',{index:String(structure.index+1).padStart(3,'0')})}${structure.representative?' ◇':''}`;}
  function renderEmbedding(){const a=state.analysis,points=a.embedding||[],svg=$('#embedding-chart');if(!points.length){svg.innerHTML=`<text x="155" y="95" text-anchor="middle" class="embedding-label">${translated('cluster.noEmbedding')}</text>`;return;}
    const minX=Math.min(...points.map(p=>p.x)),maxX=Math.max(...points.map(p=>p.x)),minY=Math.min(...points.map(p=>p.y)),maxY=Math.max(...points.map(p=>p.y));const scaleX=x=>30+(x-minX)/Math.max(maxX-minX,.0001)*255,scaleY=y=>166-(y-minY)/Math.max(maxY-minY,.0001)*145;
    svg.innerHTML=`<rect x="29" y="20" width="257" height="147" fill="#fcfdff" rx="4"/><path d="M29 166H286M30 20V166" fill="none" stroke="#e7edf6" stroke-width="1"/><path d="M30 93H286M158 20V166" fill="none" stroke="#f0f3f8" stroke-dasharray="3 4"/><text x="158" y="191" text-anchor="middle" class="embedding-label">PC 1</text><text x="10" y="94" text-anchor="middle" transform="rotate(-90 10 94)" class="embedding-label">PC 2</text>${points.map(point=>`<circle tabindex="0" role="button" aria-label="${translated('cluster.pointAria',{index:point.index+1,cluster:point.cluster})}" data-embedding="${point.index}" class="embedding-point" cx="${maxX===minX?158:scaleX(point.x)}" cy="${maxY===minY?93:scaleY(point.y)}" r="${points.length>300?2.5:points.length>80?3.5:5}" fill="${color(point.cluster)}" fill-opacity=".72"><title>${translated('cluster.pointTitle',{index:point.index+1,cluster:point.cluster})}</title></circle>`).join('')}`;
    $$('[data-embedding]',svg).forEach(point=>{point.onclick=()=>{stopPlayback();selectStructure(Number(point.dataset.embedding));};point.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();point.onclick();}};});
    const variance=a.embedding_variance_ratio||[];$('#embedding-note').textContent=variance.length>=2?t('cluster.embeddingVariance',{percent:(100*(Number(variance[0])+Number(variance[1]))).toFixed(1)}):t('cluster.embeddingNote');
  }
  function renderHeatmap(){const a=state.analysis,matrix=a.distance_matrix;if(!matrix?.length)return;const canvas=$('#distance-heatmap'),ctx=canvas.getContext('2d');const order=a.clusters.flatMap(cluster=>cluster.members),n=order.length;canvas._structureOrder=order;
    let maximum=0;for(const row of matrix)for(const v of row)maximum=Math.max(maximum,Number(v));$('#heatmap-max').textContent=`${maximum.toFixed(1)} Å`;
    canvas.width=Math.max(250,n);canvas.height=canvas.width;const size=canvas.width/n;
    for(let y=0;y<n;y++)for(let x=0;x<n;x++){const value=Number(matrix[order[y]][order[x]])/Math.max(maximum,.0001);const start=[243,247,255],middle=[65,116,217],end=[20,55,129];const t=value<.65?value/.65:(value-.65)/.35,from=value<.65?start:middle,to=value<.65?middle:end;ctx.fillStyle=`rgb(${from.map((v,i)=>Math.round(v+(to[i]-v)*t)).join(',')})`;ctx.fillRect(x*size,y*size,Math.ceil(size),Math.ceil(size));}
    let cursor=0;ctx.strokeStyle='rgba(255,255,255,.85)';ctx.lineWidth=.8;for(const cluster of a.clusters){cursor+=cluster.count;ctx.beginPath();ctx.moveTo(cursor*size,0);ctx.lineTo(cursor*size,canvas.height);ctx.moveTo(0,cursor*size);ctx.lineTo(canvas.width,cursor*size);ctx.stroke();}
  }
  async function loadViewer(){if(!state.analysis||state.fetchingViewer)return;const id=state.id,generation=state.viewerGeneration;state.fetchingViewer=true;
    const overlay=$('#viewer-loading');
    if(!window.$3Dmol){overlay.classList.add('error');overlay.innerHTML=label('viewer.componentMissing');state.fetchingViewer=false;return;}
    try{state.viewer=window.$3Dmol.createViewer($('#molecule-viewer'),{backgroundColor:'#fbfcff',antialias:true});state.viewer.setProjection('orthographic');let loaded=0;const list=state.analysis.structures;let cursor=0;
      const worker=async()=>{while(cursor<list.length){const structure=list[cursor++];const url=structure.url||`/api/targets/${encodeURIComponent(id)}/structures/${encodeURIComponent(structure.file)}?aligned=1`;const response=await fetch(url);if(!response.ok)throw new UIError('error.readStructure',{index:structure.index+1});const text=await response.text();if(id!==state.id||generation!==state.viewerGeneration)return;const model=state.viewer.addModel(text,'pdb');state.models.set(structure.index,model);loaded++;const element=$('#viewer-load-message');if(element)element.textContent=t('viewer.loadingCount',{loaded,total:list.length});}};
      await Promise.all(Array.from({length:Math.min(4,list.length)},worker));if(id!==state.id||generation!==state.viewerGeneration)return;refreshViewer();state.viewer.zoomTo();state.viewer.rotate(18,'y');state.viewer.rotate(-12,'x');state.viewer.zoom(1.5);state.viewer.render();overlay.hidden=true;state.viewer.resize();
    }catch(error){if(id===state.id&&generation===state.viewerGeneration){state.viewerError=error;overlay.hidden=false;overlay.classList.add('error');renderMessage(overlay,message('viewer.unavailable',{error}));toast(error,true);}}
    finally{if(generation===state.viewerGeneration)state.fetchingViewer=false;}
  }
  function confidenceColor(atom){const b=Number(atom.b),value=b<=1?b*100:b;return value>=90?'#315dc8':value>=70?'#65b8e4':value>=50?'#e5d376':'#d89361';}
  function refreshViewer(){if(!state.viewer||!state.analysis)return;const a=state.analysis,reps=new Set(a.clusters.map(c=>c.representative));let visible=0;
    // Dense translucent ensembles wash out; keep the same opaque style in every mode.
    for(const structure of a.structures){const model=state.models.get(structure.index);if(!model)continue;const show=(state.mode==='all'||state.mode==='representatives'&&reps.has(structure.index)||state.mode==='single'&&structure.index===state.frame)&&(!state.clusterFocus||structure.cluster===state.clusterFocus);if(!show){model.hide();continue;}model.show();visible++;const cartoon={opacity:1,thickness:.25};if(state.colorMode==='confidence')cartoon.colorfunc=confidenceColor;else cartoon.color=color(structure.cluster);model.setStyle({},{cartoon});}
    state.viewer.render();const element=$('#visible-count');if(element)element.textContent=t('viewer.visible',{visible,total:a.count})+(state.clusterFocus?' · '+t('common.cluster',{id:state.clusterFocus}):'');const caption=$('#viewer-caption');if(caption)caption.textContent=state.mode==='single'?t('viewer.singleCaption',{structure:structureLabel(analysisStructure(state.frame))}):t(state.mode==='representatives'?'viewer.repCaption':'viewer.allCaption',{count:visible});updateViewerControls();
  }
  function updateViewerControls(){if(!state.analysis||!$('#frame-slider'))return;const a=state.analysis;$$('[data-view-mode]').forEach(button=>{const selected=button.dataset.viewMode===state.mode;button.classList.toggle('active',selected);button.setAttribute('aria-selected',String(selected));});$('#frame-slider').value=state.frame;$('#frame-label').textContent=`${state.frame+1} / ${a.count}`;$('#structure-select').value=state.frame;const structure=analysisStructure(state.frame);$('#selected-info').innerHTML=state.mode==='single'&&structure?`<span>${label('cluster.column')} <strong>${structure.cluster}</strong></span><span>${label('viewer.referenceRMSD')} <strong>${Number(structure.rmsd_to_reference||0).toFixed(2)} Å</strong></span><span>pLDDT <strong>${structure.mean_plddt==null?'—':(Number(structure.mean_plddt)*100).toFixed(1)}</strong></span>`:`${label('viewer.alignedReference')}${label('viewer.representativeMarker')}`;
    const play=$('#play-button');play.dataset.i18nAriaLabel=state.playing?'viewer.pause':'viewer.play';play.setAttribute('aria-label',t(state.playing?'viewer.pause':'viewer.play'));
    $$('[data-legend-cluster]').forEach(button=>button.classList.toggle('active',Number(button.dataset.legendCluster)===state.clusterFocus));$$('[data-cluster]',$('#population-chart')).forEach(button=>button.classList.toggle('active',Number(button.dataset.cluster)===state.clusterFocus));$$('[data-embedding]',$('#embedding-chart')).forEach(point=>{const selected=state.mode==='single'&&Number(point.dataset.embedding)===state.frame;point.classList.toggle('selected',selected);point.setAttribute('fill-opacity',state.clusterFocus&&analysisStructure(point.dataset.embedding)?.cluster!==state.clusterFocus?'.15':selected?'1':'.72');});$$('[data-structure]',$('#representative-rows')).forEach(row=>row.classList.toggle('selected',state.mode==='single'&&Number(row.dataset.structure)===state.frame));
  }
  function selectStructure(index){if(!state.analysis)return;state.frame=Math.max(0,Math.min(state.analysis.count-1,index));state.mode='single';state.clusterFocus=null;refreshViewer();updateViewerControls();}
  function focusCluster(id){stopPlayback();state.clusterFocus=state.clusterFocus===id?null:id;if(state.mode==='single')state.mode='all';refreshViewer();updateViewerControls();}
  function startPlayback(){if(!state.analysis)return;stopPlayback();state.mode='single';state.clusterFocus=null;state.playing=true;const button=$('#play-button');button.textContent='Ⅱ';button.setAttribute('aria-label',t('viewer.pause'));refreshViewer();state.playTimer=setInterval(()=>{state.frame=(state.frame+1)%state.analysis.count;refreshViewer();},1000/state.fps);}

  function openDeleteTarget(id){
    const target=state.targets.find(item=>item.id===id)||(state.target?.id===id?state.target:null);if(!target)return;
    state.deleteTarget=target;setError($('#delete-error'),'');updateDeleteDialog();openDialog('delete-dialog');
  }
  function updateDeleteDialog(){
    if(!state.deleteTarget)return;$('#delete-target-name').textContent=state.deleteTarget.name;
    $('#delete-active-warning').hidden=!activeStatuses.includes(state.deleteTarget.status);
    const button=$('#delete-confirm');button.dataset.i18n=state.deleting?'delete.deleting':activeStatuses.includes(state.deleteTarget.status)?'delete.cancelAndDelete':'delete.confirm';button.textContent=t(button.dataset.i18n);
  }
  async function deleteTarget(event){
    event.preventDefault();if(state.deleting||!state.deleteTarget)return;const id=state.deleteTarget.id;state.deleting=true;setError($('#delete-error'),'');updateDeleteDialog();
    const dialog=$('#delete-dialog'),buttons=$$('button',dialog);buttons.forEach(button=>button.disabled=true);
    try{
      const current=await api(`/api/targets/${encodeURIComponent(id)}`);state.deleteTarget=current;
      if(activeStatuses.includes(current.status))state.deleteTarget=await api(`/api/targets/${encodeURIComponent(id)}/cancel`,{method:'POST',body:'{}'});
      const deadline=Date.now()+15000;
      while(true){try{await api(`/api/targets/${encodeURIComponent(id)}`,{method:'DELETE'});break;}catch(error){if(error.status===409&&error.code==='target_busy'){if(Date.now()>=deadline)throw new UIError('delete.busy');await wait(500);continue;}throw error;}}
      dialog.close();state.targets=state.targets.filter(target=>target.id!==id);if(state.id===id){location.hash='';route();}renderHome();toast(message('delete.deleted'),false,id);await refreshTargets();
    }catch(error){setError($('#delete-error'),error);}finally{state.deleting=false;buttons.forEach(button=>button.disabled=false);updateDeleteDialog();}
  }
  function addDialogLanguageControls(){
    $('#language-select').dataset.languageSelect='';
    $$('.dialog-heading').forEach(heading=>{const control=document.createElement('label');control.className='dialog-language language-control';control.innerHTML=`<span class="sr-only" data-i18n="nav.language">${translated('nav.language')}</span><select data-language-select data-i18n-aria-label="nav.language" aria-label="${translated('nav.language')}"><option value="en">English</option><option value="zh">中文</option></select>`;heading.insertBefore(control,$('.close-dialog',heading));});
    $$('[data-language-select]').forEach(select=>{select.value=i18n.language;select.addEventListener('change',()=>i18n.setLanguage(select.value));});
  }
  function updateFileLabels(){
    const json=$('#json-file').files[0];$('#json-file-label').textContent=json?json.name:t('import.pasteJSON');
    const count=$('#json-msa-files').files.length;$('#json-msa-file-label').textContent=count?t('import.filesSelected',{count}):t('submit.noFile');
  }
  function applyLocale(){
    document.title=t('app.title');i18n.apply(document);$$('[data-language-select]').forEach(select=>select.value=i18n.language);
    updateDeviceLabel();renderHome();
    if(state.target&&state.detailLoaded)updateDetailMeta(state.target);
    if(state.detailMissingError&&!state.detailLoaded&&$('#detail-missing-error'))renderMessage($('#detail-missing-error'),state.detailMissingError);
    if(state.analysis&&$('#result-area')?.children.length){updateAnalysisViews();refreshViewer();}
    // Existing input nodes, FileLists, open details, and the molecule canvas remain in place.
    $$('.draft-card').forEach(card=>{const draft=state.drafts.find(item=>item.id===card.dataset.draft);if(!draft)return;updateDraftSequence(card,draft);const drop=$('.file-drop',card);$('strong',drop).textContent=t(draft.msa_text?'submit.msaLoaded':'submit.dropMSA');$('.file-name',drop).textContent=draft.msa_name||t('submit.chooseFile');for(const kind of ['fasta','pdb']){const node=$(`[data-file-label="${kind}"]`,card);if(node)node.textContent=draft[`${kind}_name`]||t('submit.noFile');}});
    $('#draft-count').textContent=t('common.targets',{count:state.drafts.length});updateFileLabels();updateImportPreview();updateDeleteDialog();
    $$('.form-error').forEach(node=>{if(node._message)renderMessage(node,node._message);});
    $$('.toast').forEach(node=>{if(node._message)renderMessage($('.toast-text',node),node._message);});
    for(const id of ['submit-button','import-submit']){const button=$(`#${id}`);if(button.disabled)button.textContent=t('common.submitting');}
    if(state.fetchingViewer&&$('#viewer-load-message'))$('#viewer-load-message').textContent=t('viewer.loadingCount',{loaded:state.models.size,total:state.analysis?.count||0});
    if(state.viewerError&&$('#viewer-loading'))renderMessage($('#viewer-loading'),message('viewer.unavailable',{error:state.viewerError}));
  }

  function bindEvents(){
    $('#new-target-button').onclick=()=>openSubmit();$('#example-button').onclick=loadExample;$('#help-button').onclick=()=>openDialog('help-dialog');$('#import-button').onclick=()=>{setError($('#import-error'),'');openDialog('import-dialog');};$('#refresh-button').onclick=()=>{poll();loadConfig();};
    $$('.close-dialog').forEach(button=>button.onclick=()=>button.closest('dialog').close());$$('dialog').forEach(dialog=>dialog.addEventListener('click',event=>{if(event.target===dialog){const box=dialog.getBoundingClientRect();if(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom)dialog.close();}}));
    $$('.status-tabs button').forEach(button=>button.onclick=()=>{state.filter=button.dataset.filter;$$('.status-tabs button').forEach(node=>{const active=node===button;node.classList.toggle('active',active);node.setAttribute('aria-selected',String(active));});renderHome();});
    $('#search-input').addEventListener('input',event=>{state.query=event.target.value;renderHome();});
    $('#target-list').addEventListener('click',event=>{const deletion=event.target.closest('[data-delete-target]');if(deletion){event.stopPropagation();openDeleteTarget(deletion.dataset.deleteTarget);return;}const target=event.target.closest('[data-target]');if(target)location.hash=`target/${target.dataset.target}`;if(event.target.closest('[data-action="new-target"]'))openSubmit();});$('#target-list').addEventListener('keydown',event=>{if((event.key==='Enter'||event.key===' ')&&event.target.matches('[data-target]')){event.preventDefault();event.target.click();}});
    $('#add-target-button').onclick=()=>{if(state.drafts.length>=(state.config.max_targets||50)){toast(message('error.maxTargets',{maximum:state.config.max_targets||50}),true);return;}state.drafts.push(newDraft());renderDrafts();$('.dialog-scroll',$('#submit-dialog')).scrollTop=999999;};$('#draft-targets').addEventListener('click',event=>{const button=event.target.closest('.remove-draft');if(button){state.drafts=state.drafts.filter(draft=>draft.id!==button.dataset.id);renderDrafts();}});
    $('#submit-form').addEventListener('submit',submitDrafts);$('#import-form').addEventListener('submit',submitJSON);$('#delete-form').addEventListener('submit',deleteTarget);
    $('#delete-dialog').addEventListener('cancel',event=>{if(state.deleting)event.preventDefault();});
    $('#json-file').addEventListener('change',async event=>{const file=event.target.files[0];if(file){$('#json-text').value=await file.text();updateFileLabels();updateImportPreview();}});$('#json-text').addEventListener('input',updateImportPreview);$('#json-msa-files').addEventListener('change',updateFileLabels);
    $('#detail-view').addEventListener('click',event=>{const button=event.target.closest('[data-detail-action]'),action=button?.dataset.detailAction;if(action==='download'){if(button.getAttribute('aria-disabled')==='true')event.preventDefault();return;}if(action==='analysis-retry')loadAnalysis();else if(action==='delete')openDeleteTarget(state.id);else if(action)detailAction(action);});
    addDialogLanguageControls();document.addEventListener('trflow:languagechange',applyLocale);
    window.addEventListener('hashchange',route);window.addEventListener('resize',()=>{if(state.viewer){state.viewer.resize();state.viewer.render();}});document.addEventListener('visibilitychange',()=>{if(!document.hidden)poll();else stopPlayback();});
  }
  function updateImportPreview(){try{const config=JSON.parse($('#json-text').value);const count=config.samples?.length||0;$('#import-preview').textContent=count?t('import.previewCount',{count,samples:config.options?.sample_num??state.config.defaults.sample_num}):t('import.preview');}catch(error){$('#import-preview').textContent=t('import.preview');}}
  async function init(){bindEvents();applyLocale();await loadConfig();await poll();route();setInterval(poll,2000);}
  init();
})();
