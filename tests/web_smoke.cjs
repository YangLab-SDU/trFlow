/* Real-browser smoke checks. Requires Playwright and a running local web service.
 * Usage: node tests/web_smoke.cjs http://127.0.0.1:8765 outputs/web_qa
 * This script explores existing results and drafts; it does not enqueue GPU work.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { createHash } = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const {browserLaunchOptions}=require('./browser_runtime.cjs');

const base = (process.argv[2] || 'http://127.0.0.1:8765').replace(/\/$/, '');
const output = path.resolve(process.argv[3] || 'outputs/web_qa');

async function main() {
  await fs.mkdir(output, {recursive: true});
  const browser = await chromium.launch(browserLaunchOptions());
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}, deviceScaleFactor: 1});
  const page = await context.newPage();
  const errors = [];
  const remoteRequests = [];
  const mutations = [];
  // Existing service data is read-only during smoke checks. Block any
  // accidental form submission, cancellation, or deletion before it is sent.
  await page.route('**/api/**', async route => {
    const request = route.request();
    if (request.method() !== 'GET') {
      mutations.push({method: request.method(), url: request.url()});
      await route.abort('blockedbyclient');
      return;
    }
    await route.continue();
  });
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => {if (!request.url().startsWith(base) && !request.url().startsWith('data:')) remoteRequests.push(request.url());});
  try {
    await page.goto(base, {waitUntil: 'networkidle'});
    await page.locator('#target-list .target-row').first().waitFor();
    assert.match(await page.title(), /trFlow/);
    const response = await context.request.get(base + '/api/targets');
    const targets = (await response.json()).targets;
    const completed = targets.find(target => target.status === 'completed' && target.sample_num >= 3 && target.name === '2akl')
      || targets.find(target => target.status === 'completed' && target.sample_num >= 3 && target.existing)
      || targets.find(target => target.status === 'completed' && target.sample_num >= 3);
    assert.ok(completed, 'At least one completed ensemble with three structures is required');
    await page.screenshot({path: path.join(output, 'home-desktop.png'), fullPage: true});

    await page.locator('#new-target-button').click();
    const dialog = page.locator('#submit-dialog');
    await dialog.waitFor({state: 'visible'});
    const count = dialog.locator('[data-option="sample_num"]').first();
    assert.equal(await count.inputValue(), '200');
    assert.equal(await dialog.locator('.advanced-options[open]').count(), 0);
    await dialog.locator('[data-file="msa"]').first().setInputFiles({
      name: 'qa_example.a3m', mimeType: 'text/plain', buffer: Buffer.from('>query\nACDEFG\n>hit\nACDEFG\n'),
    });
    await page.waitForFunction(() => document.querySelector('[data-sequence-summary]').textContent.includes('6 residues'));
    assert.equal(await dialog.locator('[data-field="name"]').first().inputValue(), 'qa_example');
    await page.screenshot({path: path.join(output, 'submission-single-desktop.png'), fullPage: true});
    await page.locator('#add-target-button').click();
    assert.equal(await dialog.locator('.draft-card').count(), 2);
    assert.equal(await dialog.locator('[data-field="name"]').first().inputValue(), 'qa_example');
    await dialog.locator('.dialog-scroll').evaluate(element => {element.scrollTop = 0;});
    await page.screenshot({path: path.join(output, 'submission-desktop.png'), fullPage: true});
    await dialog.locator('.advanced-options summary').first().click();
    assert.equal(await dialog.locator('[data-option="seed"]').first().inputValue(), '');
    assert.equal(await dialog.locator('[data-option="geometric_exploration"]').first().isChecked(), true);
    await dialog.locator('.close-dialog').click();

    await page.locator('#import-button').click();
    await page.locator('#json-text').fill(JSON.stringify({samples: [{name: 'target_01', msa_path: 'target_01.a3m'}], options: {sample_num: 200}}, null, 2));
    await page.screenshot({path: path.join(output, 'json-import-desktop.png'), fullPage: true});
    await page.locator('#json-text').fill('{broken json');
    await page.locator('#import-submit').click();
    await page.locator('#import-error').waitFor({state: 'visible'});
    assert.match(await page.locator('#import-error').textContent(), /JSON/);
    await page.locator('#import-dialog .close-dialog').click();

    await page.locator(`.target-row[data-target="${completed.id}"]`).click();
    await page.locator('#molecule-viewer canvas').waitFor({timeout: 60000});
    await page.locator('#viewer-loading').waitFor({state: 'hidden', timeout: 60000});
    assert.equal(await page.locator('#embedding-chart .embedding-point').count(), completed.sample_num);
    assert.equal(await page.locator('#cluster-number').inputValue(), String(Math.min(10, completed.sample_num)));
    assert.match(await page.locator('#visible-count').textContent(), new RegExp(`${completed.sample_num} / ${completed.sample_num}`));
    await page.screenshot({path: path.join(output, 'ensemble-desktop.png'), fullPage: true});

    await page.locator('#cluster-number').fill('3');
    await page.locator('#cluster-number').dispatchEvent('change');
    await page.waitForFunction(() => parseInt(document.querySelector('#cluster-badge').textContent, 10) === 3);
    assert.equal(await page.locator('#representative-rows tr').count(), 3);
    assert.equal(await page.locator('#population-chart .population-column').count(), 3);
    await page.locator('[data-view-mode="representatives"]').click();
    await page.waitForFunction(() => document.querySelector('#visible-count').textContent.startsWith('3 /'));
    await page.screenshot({path: path.join(output, 'representatives-desktop.png'), fullPage: true});

    const archiveDownload = page.waitForEvent('download');
    await page.locator('#detail-actions a').click();
    const archive = await archiveDownload;
    const archiveTargetName = Array.from(String(completed.name || completed.sample_name || 'target').trim()
      .replace(/[^\p{L}\p{N}_-]+/gu, '_').replace(/^_+|_+$/g, '')).slice(0, 120).join('') || 'target';
    assert.equal(archive.suggestedFilename(), `trflow_${archiveTargetName}.zip`);
    await archive.saveAs(path.join(output, 'results.zip'));
    const archiveBytes = await fs.readFile(path.join(output, 'results.zip'));
    assert.equal(archiveBytes.subarray(0, 2).toString(), 'PK');
    assert.ok(archiveBytes.length > 1000);
    const zipInspection = spawnSync(process.env.PYTHON_EXECUTABLE || 'python', ['-c',
      `import csv,hashlib,io,json,sys,zipfile
z=zipfile.ZipFile(sys.argv[1])
names=z.namelist()
d=json.loads(z.read('clusters/clusters.json'))
structures={s['index']:s for s in d['structures']}
refs=[d['reference']['file']]
representatives=[]
metadata_errors=[]
if sorted(i for c in d['clusters'] for i in c['members'])!=sorted(structures):
    metadata_errors.append('cluster membership must cover every structure exactly once')
for s in d['structures']:
    refs.extend([s['file'],s['aligned_file']])
    if s['file']!=s['aligned_file'] or s['file']!='prediction/'+s['filename']:
        metadata_errors.append('structure paths: '+str(s['index']))
    if 'url' in s or 'original_url' in s:
        metadata_errors.append('HTTP URL retained: '+str(s['index']))
for c in d['clusters']:
    refs.extend([c['representative_file'],c['representative_prediction_file'],*c['member_files']])
    expected=[structures[i]['file'] for i in c['members']]
    if c['member_files']!=expected or c['count']!=len(c['members']):
        metadata_errors.append('cluster members: '+str(c['id']))
    if c['representative'] not in c['members'] or c['representative_prediction_file']!=structures[c['representative']]['file']:
        metadata_errors.append('cluster representative: '+str(c['id']))
    if not c['representative_file'].startswith('clusters/representatives/'):
        metadata_errors.append('representative path: '+str(c['id']))
    representatives.append({'cluster':c['id'],'path':c['representative_file'],'prediction':c['representative_prediction_file'],
        'same_aligned_bytes':z.read(c['representative_file'])==z.read(c['representative_prediction_file'])})
assignments=list(csv.DictReader(io.StringIO(z.read('clusters/assignments.csv').decode('utf-8-sig'))))
summary=list(csv.DictReader(io.StringIO(z.read('clusters/summary.csv').decode('utf-8-sig'))))
for row in assignments:
    s=structures[int(row['index'])]
    if row['file']!=s['file'] or int(row['cluster'])!=s['cluster'] or (row['is_representative'].lower() in ('true','1'))!=bool(s['representative']):
        metadata_errors.append('assignment CSV: '+row['index'])
for row in summary:
    c=next(c for c in d['clusters'] if c['id']==int(row['cluster']))
    if int(row['size'])!=c['count'] or int(row['representative_index'])!=c['representative'] or row['representative_file']!=c['representative_file'] or row['representative_prediction_file']!=c['representative_prediction_file']:
        metadata_errors.append('summary CSV: '+row['cluster'])
prediction_files=[n for n in names if n.startswith('prediction/') and n.endswith('.pdb')]
print(json.dumps({'k':d['k'],'requested_k':d['requested_k'],'sizes':[c['count'] for c in d['clusters']],
    'predictions':len(prediction_files),'representatives':representatives,'assignment_rows':len(assignments),'summary_rows':len(summary),
    'missing_refs':[p for p in refs if p not in names],'unsafe_refs':[p for p in refs if p.startswith('/') or '\\\\' in p or '..' in p.split('/')],
    'metadata_errors':metadata_errors,'old_layout':[n for n in names if n.startswith(('predictions/','aligned/','align/')) or n=='clusters.json'],
    'prediction_hashes':{n.split('/')[-1]:hashlib.sha256(z.read(n)).hexdigest() for n in prediction_files},
    'empty':[n for n in names if z.getinfo(n).file_size==0],'corrupt':z.testzip()}))`,
      path.join(output, 'results.zip')], {encoding: 'utf8'});
    assert.equal(zipInspection.status, 0, zipInspection.stderr);
    const archiveContents = JSON.parse(zipInspection.stdout);
    assert.equal(archiveContents.k, 3);
    assert.equal(archiveContents.requested_k, 3);
    assert.equal(archiveContents.predictions, completed.sample_num);
    assert.equal(archiveContents.assignment_rows, completed.sample_num);
    assert.equal(archiveContents.summary_rows, 3);
    assert.equal(archiveContents.representatives.length, 3);
    assert.equal(archiveContents.sizes.reduce((total, size) => total + size, 0), completed.sample_num);
    assert.deepEqual(archiveContents.missing_refs, []);
    assert.deepEqual(archiveContents.unsafe_refs, []);
    assert.deepEqual(archiveContents.metadata_errors, []);
    assert.deepEqual(archiveContents.old_layout, []);
    assert.ok(archiveContents.representatives.every(item => item.same_aligned_bytes));
    assert.deepEqual(archiveContents.empty, []);
    assert.equal(archiveContents.corrupt, null);
    // Verify the single prediction directory contains the displayed aligned
    // ensemble, and representatives are exact copies of those same structures.
    for (const [filename, archiveHash] of Object.entries(archiveContents.prediction_hashes)) {
      const alignedResponse = await context.request.get(`${base}/api/targets/${encodeURIComponent(completed.id)}/structures/${encodeURIComponent(filename)}?aligned=1`);
      assert.equal(alignedResponse.status(), 200);
      assert.equal(createHash('sha256').update(await alignedResponse.body()).digest('hex'), archiveHash);
    }

    await page.locator('#playback-speed').selectOption('3');
    await page.locator('#play-button').click();
    const frame = await page.locator('#frame-label').textContent();
    await page.waitForFunction(old => document.querySelector('#frame-label').textContent !== old, frame);
    assert.equal(await page.locator('[data-view-mode="single"]').getAttribute('aria-selected'), 'true');
    await page.locator('#play-button').click();
    assert.equal(await page.locator('#play-button').textContent(), '▶');

    await page.locator('#structure-select').selectOption('1');
    assert.match(await page.locator('#frame-label').textContent(), /^2 \/ /);
    await page.locator('#embedding-chart .embedding-point').first().click();
    assert.match(await page.locator('#frame-label').textContent(), /^1 \/ /);
    await page.locator('#color-mode').selectOption('confidence');
    await page.locator('#viewer-reset').click();

    const imageDownload = page.waitForEvent('download');
    await page.locator('#viewer-image').click();
    const image = await imageDownload;
    await image.saveAs(path.join(output, 'exported-ensemble.png'));
    assert.ok((await fs.stat(path.join(output, 'exported-ensemble.png'))).size > 1000);

    await page.locator('[data-view-mode="all"]').click();
    await page.setViewportSize({width: 390, height: 844});
    await page.locator('#molecule-viewer').scrollIntoViewIfNeeded();
    await page.screenshot({path: path.join(output, 'ensemble-mobile.png'), fullPage: true});
    const mobileOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
    assert.equal(mobileOverflow, false, 'Mobile page should not overflow horizontally');
    await page.locator('.breadcrumb a').click();
    await page.locator('#home-view').waitFor({state: 'visible'});
    await page.screenshot({path: path.join(output, 'home-mobile.png'), fullPage: true});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);

    assert.deepEqual(mutations, [], 'Smoke checks must not mutate the running service');
    assert.deepEqual(errors, []);
    assert.deepEqual(remoteRequests, []);
    const report = {
      base, target: completed.name, count: completed.sample_num,
      checked: ['home', 'A3M upload', 'default options', 'multi-target draft', 'invalid JSON', '3D structure',
        'interactive clustering', 'representatives', 'ZIP download', 'animation', 'PCA selection', 'pLDDT coloring', 'PNG export', 'mobile layout', 'local-only requests'],
      archiveFilename: archive.suggestedFilename(), archiveContents, mutations, errors, remoteRequests, screenshots: output,
    };
    await fs.writeFile(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
  } catch (error) {
    await page.screenshot({path: path.join(output, 'failure.png'), fullPage: true}).catch(() => {});
    throw error;
  } finally {
    await browser.close();
  }
}

main().catch(error => {console.error(error); process.exitCode = 1;});
