/* Sampling-mode UI regression with fully mocked API requests.
 * Usage: node tests/web_sampling_modes.cjs http://127.0.0.1:8765 outputs/web_sampling_qa/final
 * The service provides static assets only. Every API request is fulfilled or
 * aborted here: no target is queued, cancelled, deleted, or sent to the GPU.
 */
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const {browserLaunchOptions}=require('./browser_runtime.cjs');
const base=(process.argv[2]||'http://127.0.0.1:8765').replace(/\/$/,'');
const output=path.resolve(process.argv[3]||'outputs/web_sampling_qa/final');
const defaults={sample_num:200,geometric_exploration:true,single_step:false,
  steps:7,random_step:true,random_step_size:true,models:['Xray','NMR'],
  parallel:false,save_repr_npz:false,seed:null};
const modeOptions={single:{single_step:true,random_step:false,steps:1},
  random:{single_step:false,random_step:true,steps:7},
  fixed:{single_step:false,random_step:false}};
const uploads={msa:{name:'qa-sampling.a3m',mimeType:'text/plain',content:'>QA query\nACDEFG\n>QA homolog\nACDEFG\n'},
  fasta:{name:'qa-sampling.fasta',mimeType:'text/plain',content:'>QA query\nACDEFG\n'},
  pdb:{name:'qa-sampling.pdb',mimeType:'text/plain',content:'ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 80.00           C\nEND\n'}};

async function main(){
  assert.ok(['127.0.0.1','localhost'].includes(new URL(base).hostname),'Local static assets only');
  await fs.mkdir(output,{recursive:true});
  const browser=await chromium.launch(browserLaunchOptions());
  const context=await browser.newContext({viewport:{width:1440,height:1080},deviceScaleFactor:1});
  const page=await context.newPage();
  const checked=[],errors=[],mockRequests=[],payloads=[],unexpected=[],external=[];
  const report={base,checked,errors,mockRequests,payloads,unexpected,external,realApiRequests:0,screenshots:output};
  let configDefaults={...defaults};
  const pass=name=>{checked.push(name);console.log('PASS '+name);};
  page.on('pageerror',error=>errors.push(error.message));
  page.on('request',request=>{if(!request.url().startsWith(base)&&!request.url().startsWith('data:'))external.push(request.url());});
  await page.route('**/api/**',async route=>{
    const request=route.request(),url=new URL(request.url()),method=request.method();
    mockRequests.push(method+' '+url.pathname);
    const respond=(body,status=200)=>route.fulfill({status,contentType:'application/json',body:JSON.stringify(body)});
    if(method==='GET'&&url.pathname==='/api/config')return respond({defaults:configDefaults,max_conformations:2000,max_targets:50,device:'QA fixture'});
    if(method==='GET'&&url.pathname==='/api/targets')return respond({targets:[]});
    if(method==='POST'&&url.pathname==='/api/targets'){
      payloads.push(request.postDataJSON());
      // A controlled mock failure keeps the draft open so its retained values
      // can be checked after serialization. Nothing reaches the real service.
      return respond({error:'目标不存在',error_code:'target_not_found',error_params:{}},400);
    }
    unexpected.push(method+' '+url.pathname);await route.abort();
  });
  const dialog=page.locator('#submit-dialog');
  const cards=()=>dialog.locator('.draft-card');
  const card=()=>cards().first();
  const steps=scope=>scope.locator('[data-option="steps"]');
  const radio=(scope,mode)=>scope.locator(`[data-generation-mode="${mode}"]`);
  const translate=key=>page.evaluate(key=>window.TrFlowI18n.t(key),key);
  const locale=async language=>{
    await dialog.locator('[data-language-select]').first().selectOption(language);
    assert.equal(await page.locator('html').getAttribute('lang'),language==='zh'?'zh-CN':'en');
  };
  const expectMode=async(scope,mode,value)=>{
    const radios=scope.locator('[data-generation-mode]');
    assert.equal(await radios.count(),3);
    assert.equal(await scope.locator('[data-generation-mode]:checked').count(),1,'Exactly one generation mode is checked');
    const names=await radios.evaluateAll(nodes=>nodes.map(node=>node.name));
    assert.equal(new Set(names).size,1,'All modes share this draft\'s radio group');
    assert.ok(names[0].startsWith('generation-'),'Generation group has a per-draft prefix');
    for(const option of ['single','random','fixed']){
      assert.equal(await radio(scope,option).getAttribute('type'),'radio');
      assert.equal(await radio(scope,option).isChecked(),option===mode,`${option} checkbox state in ${mode}`);
      assert.equal(await radio(scope,option).isEnabled(),true,'Modes remain selectable');
    }
    assert.equal(await radio(scope,'single').getAttribute('data-option'),'single_step');
    assert.equal(await radio(scope,'random').getAttribute('data-option'),'random_step');
    assert.equal(await radio(scope,'fixed').getAttribute('data-option'),null);
    assert.equal(await steps(scope).isEnabled(),mode==='fixed','Only fixed mode enables its count');
    if(value!==undefined)assert.equal(await steps(scope).inputValue(),String(value),'Historical fixed count is preserved');
  };
  const choose=async(scope,mode,value)=>{await radio(scope,mode).check();await expectMode(scope,mode,value);};
  const open=async()=>{
    await page.goto(base,{waitUntil:'networkidle'});
    await page.locator('#new-target-button').click();await dialog.waitFor({state:'visible'});
    await card().locator('.advanced-options summary').click();
  };
  const prepare=async(scope,name)=>{
    await scope.locator('[data-field="name"]').fill(name);
    await scope.locator('[data-msa-mode="paste"]').click();
    await scope.locator('[data-field="msa_text"]').fill(uploads.msa.content);
    await scope.locator('[data-option="sample_num"]').fill('7');
    await scope.locator('[data-option="seed"]').fill('1234');
  };
  const submit=async()=>{
    const before=payloads.length;
    const response=page.waitForResponse(item=>item.request().method()==='POST'&&new URL(item.url()).pathname==='/api/targets');
    await page.locator('#submit-button').click();await response;
    await page.locator('#submit-error').waitFor({state:'visible'});
    await page.waitForFunction(()=>!document.querySelector('#submit-button').disabled);
    assert.equal(payloads.length,before+1,'Exactly one mocked submission');
    assert.equal(await dialog.isVisible(),true,'Mock response retains the draft');
    return payloads.at(-1).targets;
  };
  const expectPayload=(sample,mode,fixedCount)=>{
    const expected={...modeOptions[mode],...(mode==='fixed'?{steps:fixedCount}:{})};
    for(const [key,value]of Object.entries(expected))assert.equal(sample.options[key],value,`${mode} submitted ${key}`);
    assert.equal(typeof sample.options.single_step,'boolean');assert.equal(typeof sample.options.random_step,'boolean');
    assert.equal(sample.options.sample_num,7);assert.equal(sample.options.seed,1234);
    assert.deepEqual(sample.options.models,['Xray','NMR']);
    assert.equal(sample.msa_text,uploads.msa.content);
  };
  const checkEnglish=async()=>{
    const leftovers=await dialog.evaluate(node=>{
      const texts=[],walker=document.createTreeWalker(node,NodeFilter.SHOW_TEXT);
      for(let text=walker.nextNode();text;text=walker.nextNode()){
        const parent=text.parentElement;
        if(!parent||!parent.getClientRects().length||parent.closest('[data-language-select],[data-raw-diagnostic]'))continue;
        if(/[\u3400-\u9fff]/.test(text.textContent))texts.push(text.textContent.trim());
      }
      return texts;
    });assert.deepEqual(leftovers,[],'English advanced controls are fully localized');
  };
  const focusGenerationControls=async()=>dialog.locator('.dialog-scroll').evaluate(node=>{
    const group=node.querySelector('.generation-modes');
    node.scrollTop+=group.getBoundingClientRect().top-node.getBoundingClientRect().top-100;
  });
  try{
    await open();await expectMode(card(),'random',7);
    assert.equal(await card().locator('[data-option="random_step_size"]').getAttribute('type'),'checkbox');
    assert.equal(await card().locator('[data-option="random_step_size"]').isChecked(),true);
    await checkEnglish();
    pass('default random (1 or 7) mode exclusively checked with fixed count 7 disabled');

    await prepare(card(),'QA sampling');
    for(const mode of ['single','random','fixed']){
      await choose(card(),mode,7);const [sample]=await submit();expectPayload(sample,mode,7);
    }
    pass('single / random / fixed are mutually exclusive and serialize boolean flags with canonical counts 1 / 7 / typed count');

    await steps(card()).fill('11');
    for(const mode of ['single','random','fixed']){
      await choose(card(),mode,11);const [sample]=await submit();expectPayload(sample,mode,11);
      await expectMode(card(),mode,11);
    }
    pass('switching modes and serializing payloads retain the historical fixed count without mutating the draft');

    const independent=card().locator('[data-option="random_step_size"]');await independent.uncheck();
    for(const mode of ['single','random','fixed']){
      await choose(card(),mode,11);assert.equal(await independent.isChecked(),false);
      const [sample]=await submit();expectPayload(sample,mode,11);assert.equal(sample.options.random_step_size,false);
    }
    await independent.check();
    pass('random step size remains an independent checkbox in every generation mode');

    await card().locator('[data-msa-mode="upload"]').click();
    for(const [type,file]of Object.entries(uploads)){
      await card().locator(`[data-file="${type}"]`).setInputFiles({name:file.name,mimeType:file.mimeType,buffer:Buffer.from(file.content)});
    }
    await page.waitForFunction(()=>document.querySelector('[data-file-label="pdb"]').textContent==='qa-sampling.pdb');
    await page.evaluate(()=>{
      const draft=document.querySelector('.draft-card');
      window.__qaSamplingNodes={card:draft,details:draft.querySelector('details'),
        nodes:Array.from(draft.querySelectorAll('input,textarea')),
        files:Array.from(draft.querySelectorAll('[data-file]')).map(node=>({node,list:node.files,file:node.files[0]}))};
    });
    for(const mode of ['single','random','fixed']){
      await choose(card(),mode,11);
      for(const language of ['zh','en']){
        await locale(language);await expectMode(card(),mode,11);
        const retained=await page.evaluate(async()=>{
          const old=window.__qaSamplingNodes,draft=document.querySelector('.draft-card');
          return {sameCard:old.card===draft,sameDetails:old.details===draft.querySelector('details'),
            expanded:old.details.open,sameNodes:old.nodes.every(node=>node.isConnected&&draft.contains(node)),
            files:await Promise.all(old.files.map(async item=>({sameNode:item.node.isConnected,
              sameList:item.list===item.node.files,sameFile:item.file===item.node.files[0],
              count:item.node.files.length,name:item.node.files[0]?.name,content:await item.node.files[0]?.text()})))};
        });
        assert.equal(retained.sameCard,true);assert.equal(retained.sameDetails,true);
        assert.equal(retained.expanded,true);assert.equal(retained.sameNodes,true);
        for(const [index,file]of Object.values(uploads).entries())assert.deepEqual(retained.files[index],
          {sameNode:true,sameList:true,sameFile:true,count:1,name:file.name,content:file.content});
        assert.equal(await card().locator('[data-field="name"]').inputValue(),'QA sampling');
        assert.equal(await card().locator('[data-field="msa_text"]').inputValue(),uploads.msa.content);
        assert.equal(await card().locator('[data-option="sample_num"]').inputValue(),'7');
        assert.equal(await card().locator('[data-option="seed"]').inputValue(),'1234');
        for(const [mode,key]of Object.entries({single:'submit.singleStep',random:'submit.randomSteps',fixed:'submit.fixedSteps'})){
          const rendered=(await radio(card(),mode).locator('..').textContent()).trim();assert.equal(rendered,await translate(key));
        }
        if(language==='en')await checkEnglish();
      }
      const [sample]=await submit();expectPayload(sample,mode,11);
      assert.equal(sample.fasta_text,uploads.fasta.content);assert.equal(sample.init_pdb_text,uploads.pdb.content);
    }
    pass('en / zh switches preserve all modes, numeric values, open advanced panel, upload nodes, FileLists, and file bytes');

    for(const invalid of ['','0','101']){
      await choose(card(),'fixed');await steps(card()).fill(invalid);
      const before=payloads.length;await page.locator('#submit-button').click();
      await page.locator('#submit-error').waitFor({state:'visible'});
      assert.equal((await page.locator('#submit-error .message-summary').textContent()).trim(),
        await page.evaluate(()=>window.TrFlowI18n.t('error.targetSteps',{index:1})));
      assert.equal(payloads.length,before,'Invalid active fixed count cannot issue even a mocked POST');
      for(const mode of ['single','random']){
        await choose(card(),mode,invalid);const [sample]=await submit();expectPayload(sample,mode);
        await expectMode(card(),mode,invalid);
      }
      await choose(card(),'fixed',invalid);
    }
    await steps(card()).fill('100');expectPayload((await submit())[0],'fixed',100);
    await steps(card()).fill('1');expectPayload((await submit())[0],'fixed',1);
    pass('empty / zero / >100 fixed counts block only fixed mode; single/random still submit canonical counts; fixed bounds 1 and 100 work');

    await page.locator('#add-target-button').click();
    assert.equal(await cards().count(),2);
    const first=cards().nth(0),second=cards().nth(1);
    await first.locator('.advanced-options summary').click();
    await second.locator('.advanced-options summary').click();
    await prepare(second,'QA independent second');
    await choose(first,'fixed',1);await choose(second,'single',7);
    const firstName=await radio(first,'fixed').getAttribute('name'),secondName=await radio(second,'single').getAttribute('name');
    assert.notEqual(firstName,secondName,'Each draft uses a distinct native radio group');
    await expectMode(first,'fixed',1);await expectMode(second,'single',7);
    let samples=await submit();assert.equal(samples.length,2);expectPayload(samples[0],'fixed',1);expectPayload(samples[1],'single');
    await choose(first,'random',1);await choose(second,'fixed',7);await steps(second).fill('13');
    await expectMode(first,'random',1);await expectMode(second,'fixed',13);
    await locale('zh');await expectMode(first,'random',1);await expectMode(second,'fixed',13);await locale('en');
    samples=await submit();expectPayload(samples[0],'random');expectPayload(samples[1],'fixed',13);
    pass('multiple drafts have isolated radio groups, retained fixed counts, and independent submitted options');

    configDefaults={...defaults,single_step:true,random_step:true,steps:0};
    await open();await prepare(card(),'QA legacy conflict');await expectMode(card(),'single',0);
    expectPayload((await submit())[0],'single');
    configDefaults={...defaults,single_step:false,random_step:false,steps:19};
    await open();await prepare(card(),'QA configured fixed');await expectMode(card(),'fixed',19);
    expectPayload((await submit())[0],'fixed',19);
    pass('legacy both-true config normalizes to single priority; both-false config starts in fixed mode');

    // Start a clean, unsubmitted draft for readable screenshots; mock submit
    // failures above intentionally left a controlled error visible.
    await open();await prepare(card(),'QA fixed-count example');await expectMode(card(),'fixed',19);
    await focusGenerationControls();
    await page.screenshot({path:path.join(output,'fixed-mode-en.png'),fullPage:true});
    await locale('zh');await page.screenshot({path:path.join(output,'fixed-mode-zh.png'),fullPage:true});await locale('en');
    await page.setViewportSize({width:390,height:844});
    for(const language of ['en','zh']){
      await locale(language);await expectMode(card(),'fixed',19);await focusGenerationControls();
      const layout=await dialog.evaluate(node=>({width:node.clientWidth,scroll:node.scrollWidth}));
      assert.ok(layout.scroll<=layout.width+1,`${language} mobile dialog has no horizontal overflow`);
      await page.screenshot({path:path.join(output,`sampling-mobile-${language}.png`),fullPage:true});
    }
    pass('advanced generation controls remain usable without horizontal overflow on 390px mobile in both languages');

    assert.deepEqual(unexpected,[]);assert.deepEqual(external,[]);assert.deepEqual(errors,[]);
    report.passed=true;
  }catch(error){report.passed=false;report.failure=error.stack||String(error);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});throw error;}
  finally{await fs.writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
