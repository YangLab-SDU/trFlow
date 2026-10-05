/* Read-only focused regressions for delayed clustering responses and 2akl drafts.
 * Usage: node tests/web_download_focus.cjs http://127.0.0.1:8765 outputs/web_download_qa/focused
 * All non-GET API requests are blocked. Only existing completed analysis caches are re-cut.
 */
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const {browserLaunchOptions}=require('./browser_runtime.cjs');
const base=(process.argv[2]||'http://127.0.0.1:8765').replace(/\/$/,'');
const output=path.resolve(process.argv[3]||'outputs/web_download_qa/focused');

async function main(){
  await fs.mkdir(output,{recursive:true});
  const browser=await chromium.launch(browserLaunchOptions());
  const context=await browser.newContext({viewport:{width:1440,height:1000}});
  const page=await context.newPage();
  const errors=[],mutations=[],downloads=[],checked=[],plans=[],gates=[];
  const report={base,checked,errors,mutations,downloads,screenshots:output};
  const pass=name=>{checked.push(name);console.log('PASS '+name);};
  page.on('pageerror',error=>errors.push(error.message));
  page.on('request',request=>{if(new URL(request.url()).pathname.endsWith('/download'))downloads.push(request.url());});
  const targets=(await(await context.request.get(base+'/api/targets')).json()).targets;
  const target=targets.find(item=>item.name==='8CRJ_8CRI'&&item.status==='completed')
    || targets.find(item=>item.status==='completed'&&item.generated>=3);
  assert.ok(target,'An existing completed ensemble with at least three structures is required');
  report.target=target.name;report.id=target.id;
  const delayNext=k=>{
    let signalReady,release,signalDone;
    const token={k,started:false,ready:new Promise(resolve=>{signalReady=resolve;}),hold:new Promise(resolve=>{release=resolve;}),done:new Promise(resolve=>{signalDone=resolve;})};
    token.signalReady=signalReady;token.release=release;token.signalDone=signalDone;
    plans.push(token);gates.push(token);return token;
  };
  await page.route('**/api/**',async route=>{
    const request=route.request(),url=new URL(request.url());
    if(request.method()!=='GET'){mutations.push(request.method()+' '+url.pathname);await route.abort();return;}
    const index=url.pathname===`/api/targets/${target.id}/analysis`?plans.findIndex(plan=>plan.k===Number(url.searchParams.get('k'))):-1;
    if(index<0){await route.continue();return;}
    const token=plans.splice(index,1)[0];token.started=true;
    try{const response=await route.fetch();token.signalReady();await token.hold;await route.fulfill({response});}
    finally{token.signalDone();}
  });
  const ready=async k=>page.waitForFunction(expected=>{
    const link=document.querySelector('[data-detail-action="download"]');
    return document.querySelector('#cluster-number')?.value===String(expected)
      &&parseInt(document.querySelector('#cluster-badge')?.textContent,10)===expected
      &&link?.getAttribute('aria-disabled')==='false'&&link.getAttribute('href')?.endsWith('?k='+expected);
  },k);
  const settle=()=>page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  try{
    await page.goto(base,{waitUntil:'networkidle'});
    await page.locator(`.target-row[data-target="${target.id}"]`).click();
    await page.locator('#viewer-loading').waitFor({state:'hidden',timeout:60000});
    await ready(Math.min(10,target.generated));
    assert.equal(new URL(page.url()).hash,'#target/'+target.id);
    pass('normal target-row click loads real detail and enables current-k download');

    const older=delayNext(1);
    await page.locator('#cluster-number').fill('1');
    assert.equal(await page.locator('[data-detail-action="download"]').getAttribute('aria-disabled'),'true');
    assert.equal(await page.locator('[data-detail-action="download"]').getAttribute('href'),null);
    await older.ready;
    const latest=delayNext(3);
    await page.locator('#cluster-number').fill('3');
    await latest.ready;
    await page.locator('[data-detail-action="download"]').click({force:true});
    assert.deepEqual(downloads,[],'No stale download request may be sent while analysis is pending');
    latest.release();await latest.done;await ready(3);
    older.release();await older.done;await settle();await ready(3);
    assert.equal(await page.locator('#representative-rows tr').count(),3);
    await page.screenshot({path:path.join(output,'latest-k3-after-old-response.png'),fullPage:true});
    pass('out-of-order k1 response cannot overwrite newer k3 or enable stale download');

    const previousPage=delayNext(2);
    await page.locator('#cluster-number').fill('2');
    await previousPage.ready;
    await page.locator('.breadcrumb a').click();
    await page.locator('#home-view').waitFor({state:'visible'});
    await page.locator(`.target-row[data-target="${target.id}"]`).click();
    await page.locator('#viewer-loading').waitFor({state:'hidden',timeout:60000});
    const defaultK=Math.min(10,target.generated);
    await ready(defaultK);
    previousPage.release();await previousPage.done;await settle();await ready(defaultK);
    assert.equal(await page.locator('#representative-rows tr').count(),defaultK);
    pass('old-page analysis response is invalidated after home and same-target navigation');

    await page.locator('.breadcrumb a').click();
    const exampleLabel=page.locator('#example-button [data-i18n="hero.example"]');
    assert.equal((await exampleLabel.textContent()).trim(),'Load example');
    await page.locator('#language-select').selectOption('zh');
    assert.equal((await exampleLabel.textContent()).trim(),'加载示例');
    await page.locator('#language-select').selectOption('en');
    assert.equal((await exampleLabel.textContent()).trim(),'Load example');
    await page.locator('#example-button').click();
    const dialog=page.locator('#submit-dialog');
    await dialog.waitFor({state:'visible'});
    const msa=await fs.readFile(path.resolve('example/msa/2akl.a3m'),'utf8');
    assert.equal(await dialog.locator('[data-field="name"]').inputValue(),'2akl');
    assert.equal(await dialog.locator('[data-field="msa_text"]').inputValue(),msa);
    assert.equal(await dialog.locator('[data-option="sample_num"]').inputValue(),'200');
    assert.match(await dialog.locator('[data-sequence-summary]').textContent(),/116 residues/);
    assert.match(await dialog.locator('[data-sequence-summary]').textContent(),/2,414/);
    assert.equal(await page.locator('.toast-text').last().textContent(),'Example loaded. Adjust the ensemble size before submitting.');
    await page.screenshot({path:path.join(output,'2akl-unsubmitted-draft.png'),fullPage:true});
    await dialog.locator('.close-dialog').click();
    assert.deepEqual(mutations,[]);
    assert.deepEqual(errors,[]);
    pass('generic Load example / 加载示例 buttons read the exact 2akl MSA and open its draft without submitting');
    report.passed=true;
  }catch(error){report.passed=false;report.failure=error.stack||String(error);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});throw error;}
  finally{
    for(const token of gates)token.release();
    await fs.writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify(report,null,2));
    await browser.close();
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
