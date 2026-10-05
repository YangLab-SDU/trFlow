/* Bilingual UI and recoverable-delete smoke checks on an isolated QA service.
 * Start tests/test_web_delete.py --serve-fixture --port 8766 --data-dir <fresh directory>
 * with an interpreter containing NumPy/SciPy. The fixture disables the GPU worker.
 * Usage: node tests/web_ui_smoke.cjs http://127.0.0.1:8766 outputs/web_ui_qa/latest
 * API mutations are refused unless EVERY visible target matches our tiny QA fixtures.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const {browserLaunchOptions}=require('./browser_runtime.cjs');
const base = (process.argv[2] || 'http://127.0.0.1:8766').replace(/\/$/, '');
const output = path.resolve(process.argv[3] || 'outputs/web_ui_qa/latest');
const capture = `(() => {
  window.__uiQA = {models:[],viewer:null,canvas:null};
  const addModel = $3Dmol.GLViewer.prototype.addModel;
  $3Dmol.GLViewer.prototype.addModel = function(...args) {
    window.__uiQA.viewer=this;
    const model = addModel.apply(this,args), entry = {model,visible:true};
    window.__uiQA.models.push(entry);
    for (const [method,visible] of [['show',true],['hide',false]]) {
      const fn=model[method];
      model[method]=function(...values){entry.visible=visible;return fn.apply(this,values);};
    }
    return model;
  };
})();`;

async function checkEnglish(page, label) {
  const chinese = await page.evaluate(() => {
    const result=[];
    const excluded='[data-language-select],#language-select,.target-name,.file-name,#log-output,[data-raw-diagnostic]';
    const visible=element=>element.getClientRects().length && getComputedStyle(element).visibility!=='hidden';
    const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
    for(let node=walker.nextNode();node;node=walker.nextNode()) {
      const element=node.parentElement;
      if(!element || !visible(element) || element.closest(excluded))continue;
      const text=node.textContent.trim();
      if(/[\u3400-\u9fff]/.test(text))result.push(text.slice(0,160));
    }
    for(const element of document.querySelectorAll('[title],[aria-label],[placeholder],[alt]')) {
      if(!visible(element) || element.closest(excluded))continue;
      for(const name of ['title','aria-label','placeholder','alt']) {
        const value=element.getAttribute(name)||'';
        if(/[\u3400-\u9fff]/.test(value))result.push(name+': '+value);
      }
    }
    return [...new Set(result)];
  });
  assert.deepEqual(chinese,[],label+' should have fully English visible UI');
}

async function noOverflow(page,label) {
  const dimensions=await page.evaluate(()=>({viewport:innerWidth,width:document.documentElement.scrollWidth}));
  assert.ok(dimensions.width<=dimensions.viewport+1,label+' must not overflow: '+JSON.stringify(dimensions));
}

async function main() {
  assert.ok(['127.0.0.1','localhost'].includes(new URL(base).hostname),'Only local isolated fixtures may be used');
  assert.equal(new URL(base).port,'8766','The deletion smoke must not target the main 8765 service');
  await fs.mkdir(output,{recursive:true});
  const browser=await chromium.launch(browserLaunchOptions());
  const context=await browser.newContext({viewport:{width:1440,height:1000},deviceScaleFactor:1});
  const page=await context.newPage();
  const errors=[],remoteRequests=[],checked=[],recycled=new Set();
  const report={base,checked,errors,remoteRequests,screenshots:output};
  page.on('pageerror',error=>errors.push(error.message));
  page.on('request',request=>{
    if(!request.url().startsWith(base)&&!request.url().startsWith('data:'))remoteRequests.push(request.url());
  });
  await page.route('**/static/app.js',async route=>{
    const response=await route.fetch();
    await route.fulfill({response,body:capture+'\n'+await response.text()});
  });
  const pass=name=>{checked.push(name);console.log('PASS '+name);};
  const restore=async id=>{
    const response=await context.request.post(`${base}/api/recycle-bin/${id}/restore`,{data:{}});
    assert.equal(response.status(),200,await response.text());recycled.delete(id);
  };
  try {
    const response=await context.request.get(base+'/api/targets');
    assert.equal(response.status(),200);
    const targets=(await response.json()).targets;
    assert.ok(targets.length>=3,'Three fake fixtures are required');
    assert.ok(targets.every(item=>item.name.startsWith('QA ')&&item.sample_num===3&&item.length===6&&!item.existing&&item.generated<=3),
      'REFUSING UI deletion: this service contains a non-fixture target');
    const completed=targets.find(item=>item.status==='completed');
    const failed=targets.find(item=>item.status==='failed');
    const active=targets.find(item=>item.status==='queued');
    assert.ok(completed&&failed&&active,'Use a fresh fixture directory for each full run');
    report.fixtures={completed:completed.id,failed:failed.id,active:active.id};
    pass('synthetic-only mutation guard');

    await page.goto(base,{waitUntil:'networkidle'});
    await page.locator('.target-row').first().waitFor();
    assert.equal(await page.locator('#language-select').inputValue(),'en');
    assert.equal(await page.locator('html').getAttribute('lang'),'en');
    await checkEnglish(page,'Home');
    await page.screenshot({path:path.join(output,'home-en-desktop.png'),fullPage:true});
    pass('fresh English default and home labels');

    await page.locator('#language-select').selectOption('zh');
    assert.match(await page.locator('.hero h1').textContent(),/[\u3400-\u9fff]/);
    await page.reload({waitUntil:'networkidle'});
    assert.equal(await page.locator('#language-select').inputValue(),'zh');
    assert.match(await page.locator('#targets-heading').textContent(),/[\u3400-\u9fff]/);
    await page.screenshot({path:path.join(output,'home-zh-desktop.png'),fullPage:true});
    await page.locator('#language-select').selectOption('en');
    await checkEnglish(page,'Restored English home');
    pass('Chinese switch, refresh persistence and English reverse switch');

    await page.locator('#new-target-button').click();
    const dialog=page.locator('#submit-dialog');
    await dialog.waitFor({state:'visible'});
    await dialog.locator('[data-file="msa"]').first().setInputFiles({
      name:'qa_language_draft.a3m',mimeType:'text/plain',buffer:Buffer.from('>query\nACDEFG\n>hit\nACDEFG\n'),
    });
    await dialog.locator('[data-field="name"]').first().fill('Unsubmitted QA draft');
    await dialog.locator('[data-msa-mode="paste"]').first().click();
    const msa='>query\nACDEFG\n>unsubmitted_hit\nACDEFG\n';
    await dialog.locator('[data-field="msa_text"]').first().fill(msa);
    await dialog.locator('[data-option="sample_num"]').first().fill('7');
    await dialog.locator('.advanced-options summary').first().click();
    await dialog.locator('[data-option="seed"]').first().fill('1234');
    await dialog.locator('[data-model="NMR"]').first().uncheck();
    await page.evaluate(()=>{
      window.__uiQA.draftMsa=document.querySelector('#submit-dialog [data-field="msa_text"]');
      window.__uiQA.draftFile=document.querySelector('#submit-dialog [data-file="msa"]');
    });
    await checkEnglish(page,'Submission dialog');
    await dialog.locator('[data-language-select]').selectOption('zh');
    assert.match(await dialog.locator('h2').textContent(),/[\u3400-\u9fff]/);
    await dialog.locator('[data-language-select]').selectOption('en');
    await checkEnglish(page,'Submission dialog after reverse switch');
    assert.equal(await dialog.locator('[data-field="name"]').first().inputValue(),'Unsubmitted QA draft');
    assert.equal(await dialog.locator('[data-field="msa_text"]').first().inputValue(),msa);
    assert.equal(await dialog.locator('[data-option="sample_num"]').first().inputValue(),'7');
    assert.equal(await dialog.locator('[data-option="seed"]').first().inputValue(),'1234');
    assert.equal(await dialog.locator('[data-model="NMR"]').first().isChecked(),false);
    assert.equal(await dialog.locator('.advanced-options[open]').count(),1);
    assert.ok(await page.evaluate(()=>window.__uiQA.draftMsa===document.querySelector('#submit-dialog [data-field="msa_text"]')&&
      window.__uiQA.draftFile===document.querySelector('#submit-dialog [data-file="msa"]')&&window.__uiQA.draftFile.files[0].name==='qa_language_draft.a3m'));
    await page.screenshot({path:path.join(output,'draft-language-preserved.png'),fullPage:true});
    await dialog.locator('.close-dialog').click();
    pass('modal language switch preserves unsubmitted draft, options and file nodes');

    await page.locator('#import-button').click();
    const importDialog=page.locator('#import-dialog');
    await importDialog.locator('#json-text').fill('{invalid JSON');
    await importDialog.locator('#import-submit').click();
    await importDialog.locator('#import-error').waitFor({state:'visible'});
    await checkEnglish(page,'Import validation');
    await importDialog.locator('[data-language-select]').selectOption('zh');
    assert.match(await importDialog.locator('#import-error').textContent(),/[\u3400-\u9fff]/);
    assert.equal(await importDialog.locator('#json-text').inputValue(),'{invalid JSON');
    await importDialog.locator('[data-language-select]').selectOption('en');
    await checkEnglish(page,'Import reverse switch');
    await importDialog.locator('.close-dialog').click();
    await page.locator('#help-button').click();
    await checkEnglish(page,'Help dialog');
    await page.locator('#help-dialog [data-language-select]').selectOption('zh');
    assert.match(await page.locator('#help-dialog h2').textContent(),/[\u3400-\u9fff]/);
    await page.locator('#help-dialog [data-language-select]').selectOption('en');
    await page.locator('#help-dialog .dialog-heading .close-dialog').click();
    pass('JSON validation and help dialogs translate without replacing input');

    await page.locator(`.target-row[data-target="${completed.id}"]`).click();
    await page.locator('#molecule-viewer canvas').waitFor({timeout:60000});
    await page.locator('#viewer-loading').waitFor({state:'hidden',timeout:60000});
    await checkEnglish(page,'Completed detail');
    let modelStyles=await page.evaluate(()=>window.__uiQA.models.map(entry=>({visible:entry.visible,opacity:entry.model.selectedAtoms({})[0]?.style?.cartoon?.opacity})));
    assert.equal(modelStyles.length,3);
    assert.ok(modelStyles.every(item=>item.visible&&item.opacity===1));
    await page.locator('#cluster-number').fill('2');
    await page.locator('#cluster-number').dispatchEvent('change');
    await page.waitForFunction(()=>parseInt(document.querySelector('#cluster-badge').textContent,10)===2);
    await page.locator('[data-view-mode="single"]').click();
    await page.locator('#structure-select').selectOption('1');
    await page.locator('#color-mode').selectOption('confidence');
    const viewerBox=await page.locator('#molecule-viewer').boundingBox();
    await page.mouse.move(viewerBox.x+viewerBox.width*.45,viewerBox.y+viewerBox.height*.5);
    await page.mouse.down();
    await page.mouse.move(viewerBox.x+viewerBox.width*.7,viewerBox.y+viewerBox.height*.62,{steps:12});
    await page.mouse.up();
    const camera=await page.evaluate(()=>{
      window.__uiQA.canvas=document.querySelector('#molecule-viewer canvas');return window.__uiQA.viewer.getView();
    });
    await page.locator('#language-select').selectOption('zh');
    assert.match(await page.locator('.viewer-panel h2').textContent(),/[\u3400-\u9fff]/);
    assert.equal(await page.locator('#cluster-panel .range-labels > span').first().textContent(),'1 类');
    assert.equal(await page.locator('#cluster-max-label').textContent(),'3 类');
    await page.locator('#language-select').selectOption('en');
    await checkEnglish(page,'Completed detail after reverse switch');
    assert.equal(await page.locator('#cluster-panel .range-labels > span').first().textContent(),'1 cluster');
    assert.equal(await page.locator('#cluster-max-label').textContent(),'3 clusters');
    assert.equal(await page.locator('#cluster-number').inputValue(),'2');
    assert.equal(await page.locator('[data-view-mode="single"]').getAttribute('aria-selected'),'true');
    assert.equal(await page.locator('#structure-select').inputValue(),'1');
    assert.equal(await page.locator('#frame-slider').inputValue(),'1');
    assert.equal(await page.locator('#color-mode').inputValue(),'confidence');
    assert.deepEqual(await page.evaluate(()=>window.__uiQA.viewer.getView()),camera);
    assert.ok(await page.evaluate(()=>window.__uiQA.canvas===document.querySelector('#molecule-viewer canvas')));
    modelStyles=await page.evaluate(()=>window.__uiQA.models.filter(entry=>entry.visible).map(entry=>entry.model.selectedAtoms({})[0]?.style?.cartoon?.opacity));
    assert.deepEqual(modelStyles,[1]);
    await page.screenshot({path:path.join(output,'detail-language-preserved.png'),fullPage:true});
    pass('viewer camera, canvas, mode, frame, K and coloring survive language switch');
    pass('cluster range endpoints translate both ways, including English singular');
    pass('all and single models retain opacity 1');

    await page.setViewportSize({width:390,height:844});
    await page.locator('#molecule-viewer').scrollIntoViewIfNeeded();
    await noOverflow(page,'Mobile detail');
    await page.screenshot({path:path.join(output,'detail-en-mobile.png'),fullPage:true});
    await page.locator('.breadcrumb a').click();
    await page.locator('#home-view').waitFor({state:'visible'});
    await noOverflow(page,'Mobile home');
    await page.screenshot({path:path.join(output,'home-en-mobile.png'),fullPage:true});
    await page.locator('#new-target-button').click();
    await noOverflow(page,'Mobile submission');
    await page.screenshot({path:path.join(output,'submission-en-mobile.png'),fullPage:true});
    await page.locator('#submit-dialog .close-dialog').click();
    await page.setViewportSize({width:1440,height:1000});
    pass('390px home, detail and submission layout without horizontal overflow');

    const row=page.locator(`.target-row[data-target="${completed.id}"]`);
    await row.locator('[data-delete-target]').click();
    assert.equal(new URL(page.url()).hash,'','Delete icon must not navigate the row');
    await page.locator('#delete-dialog').waitFor({state:'visible'});
    await checkEnglish(page,'Delete confirmation');
    await page.screenshot({path:path.join(output,'delete-confirm-en.png'),fullPage:true});
    await page.locator('#delete-dialog .dialog-heading .close-dialog').click();
    assert.equal((await context.request.get(`${base}/api/targets/${completed.id}`)).status(),200);
    await row.locator('[data-delete-target]').click();
    recycled.add(completed.id);
    const deleteResponse=page.waitForResponse(response=>response.request().method()==='DELETE'&&response.url()===`${base}/api/targets/${completed.id}`);
    await page.locator('#delete-confirm').click();
    assert.equal((await deleteResponse).status(),200);
    await row.waitFor({state:'detached'});
    assert.equal((await context.request.get(`${base}/api/targets/${completed.id}`)).status(),404);
    const undoResponse=page.waitForResponse(response=>response.request().method()==='POST'&&response.url().endsWith(`/recycle-bin/${completed.id}/restore`));
    await page.locator('.toast-undo').last().click();
    assert.equal((await undoResponse).status(),200);recycled.delete(completed.id);
    await page.locator(`.target-row[data-target="${completed.id}"]`).waitFor();
    pass('home delete icon, confirmation cancellation, deletion and Undo');

    await page.locator(`.target-row[data-target="${failed.id}"]`).click();
    await page.locator('[data-detail-action="delete"]').click();
    await page.locator('#delete-dialog [data-language-select]').selectOption('zh');
    assert.match(await page.locator('#delete-dialog h2').textContent(),/[\u3400-\u9fff]/);
    await page.screenshot({path:path.join(output,'delete-confirm-zh.png'),fullPage:true});
    await page.locator('#delete-dialog [data-language-select]').selectOption('en');
    await checkEnglish(page,'Failed-target delete confirmation');
    recycled.add(failed.id);
    const failedDelete=page.waitForResponse(response=>response.request().method()==='DELETE'&&response.url()===`${base}/api/targets/${failed.id}`);
    await page.locator('#delete-confirm').click();
    assert.equal((await failedDelete).status(),200);
    await page.locator('#home-view').waitFor({state:'visible'});
    assert.equal(new URL(page.url()).hash,'');
    await restore(failed.id);
    await page.locator('#refresh-button').click();
    await page.locator(`.target-row[data-target="${failed.id}"]`).waitFor();
    pass('detail deletion returns home; both confirmation locales work');

    const operations=[];
    const recordOperation=request=>{
      if(request.url().includes(`/targets/${active.id}`)&&['POST','DELETE'].includes(request.method()))operations.push(request.method()+' '+new URL(request.url()).pathname);
    };
    page.on('request',recordOperation);
    await page.locator(`.target-row[data-target="${active.id}"] [data-delete-target]`).click();
    assert.match(await page.locator('#delete-confirm').textContent(),/cancel/i);
    recycled.add(active.id);
    const activeDelete=page.waitForResponse(response=>response.request().method()==='DELETE'&&response.url()===`${base}/api/targets/${active.id}`);
    await page.locator('#delete-confirm').click();
    assert.equal((await activeDelete).status(),200);
    assert.deepEqual(operations,[`POST /api/targets/${active.id}/cancel`,`DELETE /api/targets/${active.id}`]);
    page.off('request',recordOperation);
    await restore(active.id);
    assert.equal((await (await context.request.get(`${base}/api/targets/${active.id}`)).json()).status,'cancelled');
    pass('active target explicitly cancels before deletion; restore does not restart work');

    const svg=await context.newPage();
    await svg.emulateMedia({reducedMotion:'no-preference'});
    await svg.goto(base+'/static/ensemble-hero.svg');
    assert.equal(await svg.locator('.moving-conformation').count(),1);
    assert.ok(await svg.locator('.moving-conformation').evaluate(node=>getComputedStyle(node).display!=='none'));
    assert.ok(await svg.locator('.static-conformation').evaluate(node=>getComputedStyle(node).display==='none'));
    assert.equal(await svg.locator('animateTransform').getAttribute('dur'),'9s');
    const initialMatrix=await svg.locator('.moving-conformation > g').evaluate(node=>{const matrix=node.getCTM();return {a:matrix.a,b:matrix.b};});
    await svg.waitForFunction(before=>{
      const matrix=document.querySelector('.moving-conformation > g').getCTM();return Math.abs(matrix.a-before.a)+Math.abs(matrix.b-before.b)>.003;
    },initialMatrix,{timeout:6000});
    await svg.screenshot({path:path.join(output,'hero-animated.png')});
    await svg.emulateMedia({reducedMotion:'reduce'});
    assert.ok(await svg.locator('.moving-conformation').evaluate(node=>getComputedStyle(node).display==='none'));
    assert.ok(await svg.locator('.static-conformation').evaluate(node=>getComputedStyle(node).display!=='none'));
    await svg.screenshot({path:path.join(output,'hero-reduced-motion.png')});
    await svg.close();
    pass('hero contains live hinge animation and reduced-motion static alternative');

    assert.deepEqual(errors,[]);
    assert.deepEqual(remoteRequests,[]);
    pass('no JavaScript errors or external network requests');
    report.passed=true;
  } catch(error) {
    report.passed=false;report.failure=error.stack||String(error);
    await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});
    throw error;
  } finally {
    for(const id of recycled) {
      try {await restore(id);} catch(error) {report.cleanupFailure=String(error);}
    }
    await fs.writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify(report,null,2));
    await browser.close();
  }
}

main().catch(error=>{console.error(error);process.exitCode=1;});
