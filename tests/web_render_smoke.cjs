/* Inspect real completed ensembles. No submissions, GPU runs, or clustering changes. */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const {browserLaunchOptions}=require('./browser_runtime.cjs');
const base = process.argv[2] || 'http://127.0.0.1:8765';
const output = path.resolve(process.argv[3] || 'outputs/web_render_qa');
const capture = `(() => {
  window.__renderQA = [];
  const original = $3Dmol.GLViewer.prototype.addModel;
  $3Dmol.GLViewer.prototype.addModel = function(...args) {
    const model = original.apply(this, args), record = {model, visible:true};
    window.__renderQA.push(record);
    for (const [method, visible] of [['show',true],['hide',false]]) {
      const fn = model[method];
      model[method] = function(...values) {record.visible=visible;return fn.apply(this,values);};
    }
    return model;
  };
})();`;

(async () => {
  await fs.mkdir(output, {recursive:true});
  const browser = await chromium.launch(browserLaunchOptions());
  const page = await browser.newPage({viewport:{width:1440,height:1000}});
  const errors = [];
  const mutations=[];
  await page.route('**/api/**',async route=>{
    const request=route.request();
    if(request.method()!=='GET'){
      mutations.push({method:request.method(),url:request.url()});
      return route.abort('blockedbyclient');
    }
    return route.continue();
  });
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/static/app.js', async route => {
    const response = await route.fetch();
    await route.fulfill({response, body:capture + '\n' + await response.text()});
  });
  try {
    const targets = (await (await page.request.get(base + '/api/targets')).json()).targets;
    const target = targets.find(item => item.status==='completed' && item.generated===200);
    assert.ok(target, 'A real completed 200-conformation ensemble is required');
    await page.goto(base + '/#target/' + target.id);
    await page.locator('#molecule-viewer canvas').waitFor({timeout:120000});
    await page.locator('#viewer-loading').waitFor({state:'hidden',timeout:120000});
    assert.match(await page.locator('#visible-count').textContent(), /^200 \/ 200/);
    const styles = () => page.evaluate(() => window.__renderQA.map(entry => {
      const cartoon=entry.model.selectedAtoms({})[0].style.cartoon;
      return {visible:entry.visible,opacity:cartoon.opacity,color:cartoon.color,confidence:typeof cartoon.colorfunc==='function'};
    }));
    let inspected = await styles();
    assert.equal(inspected.length, 200);
    assert.ok(inspected.every(entry => entry.opacity===1 && entry.visible));
    await page.locator('.viewer-panel').screenshot({path:path.join(output,'all-200.png')});
    await page.locator('[data-view-mode="representatives"]').click();
    inspected = await styles();
    assert.equal(inspected.filter(entry => entry.visible).length, Number(await page.locator('#cluster-number').inputValue()));
    assert.ok(inspected.filter(entry=>entry.visible).every(entry=>entry.opacity===1));
    await page.locator('.viewer-panel').screenshot({path:path.join(output,'representatives.png')});
    await page.locator('[data-view-mode="all"]').click();
    await page.locator('#color-mode').selectOption('confidence');
    inspected = await styles();
    assert.ok(inspected.every(entry => entry.visible && entry.opacity===1 && entry.confidence));
    await page.locator('.viewer-panel').screenshot({path:path.join(output,'all-200-confidence.png')});
    await page.locator('[data-view-mode="single"]').click();
    assert.equal((await styles()).filter(entry=>entry.visible).length, 1);
    await page.locator('#play-button').click();
    const before = await page.locator('#frame-label').textContent();
    await page.waitForFunction(previous => document.querySelector('#frame-label').textContent!==previous, before);
    await page.locator('#play-button').click();
    assert.equal((await styles()).filter(entry=>entry.visible).length, 1);
    assert.deepEqual(errors, []);
    assert.deepEqual(mutations,[]);
    const report={target:target.name,id:target.id,real_predictions:200,all_visible:200,all_opacity:1,representative_opacity:1,single_and_playback:true,cluster_and_confidence_colors:true,errors,mutations};
    await fs.writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify(report,null,2));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
