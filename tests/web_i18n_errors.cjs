/* Controlled-error bilingual checks with entirely mocked API fixtures.
 * Usage: node tests/web_i18n_errors.cjs http://127.0.0.1:8765 outputs/web_i18n_qa/final
 * The real service supplies static files ONLY. No real API request, prediction,
 * submission, cancellation, or deletion is allowed through the route barrier.
 */
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const {browserLaunchOptions}=require('./browser_runtime.cjs');
const base=(process.argv[2]||'http://127.0.0.1:8765').replace(/\/$/,'');
const output=path.resolve(process.argv[3]||'outputs/web_i18n_qa/final');
const interrupted='服务已中断；可重新提交此目标。';
const raw='RuntimeError: 模型原始诊断 — tensor shape mismatch\n  at predictor.py:42';
const options={sample_num:3,models:['NMR'],geometric_exploration:false,single_step:false,steps:7,seed:null};
const target=(id,name,extra={})=>({id,name,status:'failed',stage:1,length:6,sample_num:3,generated:0,
  options,created_at:'2026-10-04T00:00:00Z',started_at:'2026-10-04T00:00:01Z',
  finished_at:'2026-10-04T00:00:02Z',existing:false,error:null,log_tail:raw,...extra});
const fixtures=[
  target('qa-legacy','QA legacy interrupted',{error:interrupted}),
  target('qa-structured','QA structured failure',{generated:2,error:'仅生成 2/3 个构象',
    error_code:'incomplete_predictions',error_params:{generated:2,expected:3}}),
  target('qa-raw','QA 用户原文',{error:raw}),
  target('qa-wrapped','QA wrapped diagnostic',{error:'结构分析失败：'+raw,
    error_code:'analysis_failed',error_params:{detail:raw}}),
  target('qa-analysis','QA analysis error',{status:'completed',stage:5,generated:3}),
];
const failure=(code,params={},error='受控应用错误')=>({error,error_code:code,error_params:params});

async function checkEnglish(page,label){
  const leftovers=await page.evaluate(names=>{
    const excluded='[data-language-select],#language-select,[data-raw-diagnostic],#log-output';
    const visible=element=>element.getClientRects().length&&getComputedStyle(element).visibility!=='hidden';
    const clean=value=>names.reduce((text,name)=>text.split(name).join(''),value);
    const found=[],walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
    for(let node=walker.nextNode();node;node=walker.nextNode()){
      const element=node.parentElement;if(!element||!visible(element)||element.closest(excluded))continue;
      const value=clean(node.textContent.trim());if(/[\u3400-\u9fff]/.test(value))found.push(value.slice(0,160));
    }
    for(const element of document.querySelectorAll('[title],[aria-label],[placeholder],[alt]')){
      if(!visible(element)||element.closest(excluded))continue;
      for(const attribute of ['title','aria-label','placeholder','alt']){
        const value=clean(element.getAttribute(attribute)||'');if(/[\u3400-\u9fff]/.test(value))found.push(attribute+': '+value);
      }
    }
    return [...new Set(found)];
  },fixtures.map(item=>item.name));
  assert.deepEqual(leftovers,[],label+': controlled English UI must not contain Chinese');
}

async function main(){
  assert.ok(['127.0.0.1','localhost'].includes(new URL(base).hostname),'Local static files only');
  await fs.mkdir(output,{recursive:true});
  const browser=await chromium.launch(browserLaunchOptions());
  const context=await browser.newContext({viewport:{width:1440,height:1000},deviceScaleFactor:1});
  const page=await context.newPage();
  const checked=[],errors=[],mockRequests=[],unexpected=[],external=[];
  const report={base,checked,errors,mockRequests,unexpected,external,realApiRequests:0,screenshots:output};
  let exampleMode='coded';
  const pass=name=>{checked.push(name);console.log('PASS '+name);};
  page.on('pageerror',error=>errors.push(error.message));
  page.on('request',request=>{if(!request.url().startsWith(base)&&!request.url().startsWith('data:'))external.push(request.url());});
  await page.route('**/api/**',async route=>{
    const request=route.request(),url=new URL(request.url()),method=request.method();
    mockRequests.push(method+' '+url.pathname);
    const respond=(body,status=200)=>route.fulfill({status,contentType:'application/json',body:JSON.stringify(body)});
    if(method==='GET'&&url.pathname==='/api/config')return respond({defaults:options,max_conformations:2000,max_targets:50,device:'QA fixture'});
    if(method==='GET'&&url.pathname==='/api/targets')return respond({targets:fixtures});
    if(method==='GET'&&url.pathname==='/api/example'){
      if(exampleMode==='network')return route.abort('connectionfailed');
      return respond(failure('target_not_found',{},'目标不存在'),404);
    }
    if(method==='POST'&&url.pathname==='/api/targets')return respond(failure('incomplete_predictions',{generated:2,expected:3},'仅生成 2/3 个构象'),400);
    if(method==='GET'&&url.pathname==='/api/targets/qa-analysis/analysis')return respond(failure('analysis_timeout',{},'结构分析超时，可从详情页重试'),500);
    if(method==='GET'&&/^\/api\/targets\/[^/]+$/.test(url.pathname)){
      const item=fixtures.find(item=>url.pathname.endsWith('/'+item.id));
      return item?respond(item):respond(failure('target_not_found',{},'目标不存在'),404);
    }
    unexpected.push(method+' '+url.pathname);await route.abort();
  });
  const text=(key,parameters={})=>page.evaluate(({key,parameters})=>window.TrFlowI18n.t(key,parameters),{key,parameters});
  const switchLocale=async value=>{
    const dialog=page.locator('dialog[open] [data-language-select]');
    await (await dialog.count()?dialog.first():page.locator('#language-select')).selectOption(value);
    assert.equal(await page.locator('html').getAttribute('lang'),value==='zh'?'zh-CN':'en');
  };
  const expectText=async(selector,key,parameters={})=>assert.equal((await page.locator(selector).textContent()).trim(),await text(key,parameters));
  const bilingual=async(selector,key,parameters={})=>{
    await switchLocale('en');await expectText(selector,key,parameters);
    const english=(await page.locator(selector).textContent()).trim();
    assert.ok(!/[\u3400-\u9fff]/.test(english),'Controlled English error cannot contain Chinese');
    assert.ok(!english.startsWith('server.'),'No untranslated error key');
    await switchLocale('zh');await expectText(selector,key,parameters);
    assert.ok(/[\u3400-\u9fff]/.test(await page.locator(selector).textContent()),'Controlled Chinese error must be localized');
    await switchLocale('en');await expectText(selector,key,parameters);
    assert.equal((await page.locator(selector).textContent()).trim(),english);
  };
  const home=async()=>{await page.goto(base,{waitUntil:'networkidle'});await page.locator('.target-row').first().waitFor();};
  const open=async id=>{await page.locator(`.target-row[data-target="${id}"]`).click();await page.locator('#detail-error').waitFor({state:'attached'});};
  try{
    await home();
    const catalog=await page.evaluate(()=>window.TrFlowI18n.messages);
    assert.deepEqual(Object.keys(catalog.en).sort(),Object.keys(catalog.zh).sort());
    for(const [key,english]of Object.entries(catalog.en)){
      assert.ok(!/[\u3400-\u9fff]/.test(english),'English catalog: '+key);
      assert.ok(catalog.zh[key],'Missing Chinese: '+key);
      const placeholders=value=>[...value.matchAll(/\{(\w+)\}/g)].map(match=>match[1]).sort();
      assert.deepEqual(placeholders(english),placeholders(catalog.zh[key]),'Translation parameters: '+key);
    }
    for(const code of ['service_interrupted','incomplete_predictions','target_not_found','analysis_timeout'])assert.ok(catalog.en['server.'+code],'Missing controlled-error translation: '+code);
    await checkEnglish(page,'Home');
    assert.equal(await page.locator('.target-icon svg.protein-conformations').count(),fixtures.length);
    assert.equal(await page.locator('.target-icon svg .ribbon-beta').count(),fixtures.length);
    await page.locator('.target-icon').first().screenshot({path:path.join(output,'task-ribbon-icon.png')});
    pass('bilingual catalog is complete and task icons use protein ribbon variants');

    await open('qa-legacy');
    await bilingual('#detail-error .message-summary','server.service_interrupted');
    await page.locator('#detail-log summary').click();
    assert.equal(await page.locator('#log-output').textContent(),raw);
    await checkEnglish(page,'Legacy failed page');
    await page.screenshot({path:path.join(output,'legacy-failure-en.png'),fullPage:true});
    await switchLocale('zh');
    await expectText('#detail-error .message-summary','server.service_interrupted');
    assert.equal(await page.locator('#log-output').textContent(),raw);
    await page.screenshot({path:path.join(output,'legacy-failure-zh.png'),fullPage:true});
    await page.locator('.running-illustration').screenshot({path:path.join(output,'failure-ribbon-icon.png')});
    pass('persisted legacy interruption switches en / zh / en while model log bytes remain unchanged');

    await home();await open('qa-structured');
    await bilingual('#detail-error .message-summary','server.incomplete_predictions',{generated:2,expected:3});
    await checkEnglish(page,'Structured failed page');
    pass('structured failure codes and numeric parameters localize in both languages');

    await home();await open('qa-raw');
    assert.equal(await page.locator('#detail-view h1').textContent(),'QA 用户原文');
    const diagnostic=page.locator('#detail-error [data-raw-diagnostic]');
    await diagnostic.waitFor({state:'attached'});
    for(const language of ['en','zh','en']){
      await switchLocale(language);
      assert.equal(await diagnostic.textContent(),raw);
      assert.equal(await page.locator('#log-output').textContent(),raw);
      assert.equal(await page.locator('#detail-view h1').textContent(),'QA 用户原文');
    }
    await checkEnglish(page,'Marked original diagnostic page');
    await home();await open('qa-wrapped');
    const wrapped=page.locator('#detail-error [data-raw-diagnostic]');await wrapped.waitFor({state:'attached'});
    for(const language of ['en','zh','en']){
      await switchLocale(language);assert.equal(await wrapped.textContent(),raw);
      const heading=(await text('server.analysis_failed',{detail:''})).trim();
      assert.ok((await page.locator('#detail-error').textContent()).includes(heading),'Analysis-failure UI heading must localize');
    }
    await checkEnglish(page,'Controlled failure wrapper with marked original diagnostic');
    pass('unknown and coded model diagnostics, trace lines, and user names retain original wording');

    await home();await switchLocale('en');await page.locator('#example-button').click();
    await page.locator('.toast.error .toast-text').last().waitFor();
    await bilingual('.toast.error .toast-text .message-summary','server.target_not_found');
    await checkEnglish(page,'Controlled API error toast');
    pass('coded read-only API error toast re-translates without retaining a raw Chinese snapshot');

    await home();await page.evaluate(()=>{location.hash='target/qa-not-found';});
    await page.locator('#detail-missing-error').waitFor();
    await bilingual('#detail-missing-error .message-summary','server.target_not_found');
    await checkEnglish(page,'Missing-target page');
    pass('missing-target 404 page preserves structured error state across language switches');

    await home();await open('qa-analysis');
    await page.locator('[data-detail-action="analysis-retry"]').waitFor();
    await bilingual('#running-description .message-summary','server.analysis_timeout');
    for(const language of ['zh','en']){
      await switchLocale(language);
      const reason=await text('server.analysis_timeout');
      assert.equal(await page.locator('.toast.error .toast-text').last().textContent(),await text('notice.analysisError',{error:reason}));
      await expectText('#running-description .message-summary','server.analysis_timeout');
    }
    await checkEnglish(page,'Analysis failure page and nested toast');
    pass('analysis failure description and nested toast reason both re-translate live');

    await home();await switchLocale('en');await page.locator('#new-target-button').click();
    const dialog=page.locator('#submit-dialog');await dialog.waitFor({state:'visible'});
    await dialog.locator('[data-field="name"]').fill('QA 用户草稿');
    await dialog.locator('[data-msa-mode="paste"]').click();
    await dialog.locator('[data-field="msa_text"]').fill('>QA query\nACDEFG\n');
    await dialog.locator('[data-option="sample_num"]').fill('3');
    await page.locator('#submit-button').click();
    await page.locator('#submit-error').waitFor({state:'visible'});
    await bilingual('#submit-error .message-summary','server.incomplete_predictions',{generated:2,expected:3});
    assert.equal(await dialog.locator('[data-field="name"]').inputValue(),'QA 用户草稿');
    assert.equal(await dialog.locator('[data-field="msa_text"]').inputValue(),'>QA query\nACDEFG\n');
    await checkEnglish(page,'Controlled form API error');
    await dialog.locator('.close-dialog').click();
    pass('mocked submission API error localizes without changing the unsubmitted draft');

    exampleMode='network';await home();await switchLocale('en');await page.locator('#example-button').click();
    const network=page.locator('.toast.error .toast-text').last();await network.waitFor();
    const english=await network.textContent();
    assert.ok(!/[\u3400-\u9fff]/.test(english));
    await switchLocale('zh');assert.ok(/[\u3400-\u9fff]/.test(await network.textContent()));
    assert.match(await network.textContent(),/Failed to fetch/,'Browser diagnostic must retain its original wording');
    await switchLocale('en');assert.equal(await network.textContent(),english);
    await checkEnglish(page,'Network error toast');
    pass('network error UI prefix localizes while the native browser diagnostic is preserved');

    assert.deepEqual(unexpected,[]);assert.deepEqual(external,[]);assert.deepEqual(errors,[]);
    report.catalogKeys=Object.keys(catalog.en).length;report.passed=true;
  }catch(error){report.passed=false;report.failure=error.stack||String(error);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});throw error;}
  finally{await fs.writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
