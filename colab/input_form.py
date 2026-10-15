"""Dependency-free, conditional Colab input card and validated form settings.

The Promise expression mounts the markup in the current Colab cell's output
frame. Python waits with google.colab.output.eval_js(timeout_sec=None).
Uploads continue through Colab's supported files.upload() after submission;
no sequence or file is sent to a third party by this browser-side form.
"""
import html
import json
import re
import uuid


INPUT_MODES = ("Paste sequence", "Upload FASTA", "Upload A3M")
TIMEOUT_CHOICES = (10, 20, 30, 60)
DEFAULT_SETTINGS = {"mode": "Paste sequence", "sequence": "", "allow_public": False,
                    "use_env": True, "use_filter": True, "timeout_minutes": 20}


def validate_settings(value):
    if not isinstance(value, dict):
        raise ValueError("Input selection did not finish. Rerun Step 2.")
    if value.get("mode") not in INPUT_MODES:
        raise ValueError("Choose Paste sequence, Upload FASTA, or Upload A3M.")
    if not isinstance(value.get("sequence"), str) or len(value["sequence"].encode("utf-8")) > 2 * 1024**2:
        raise ValueError("The sequence input is invalid or exceeds 2 MiB.")
    for name in ("allow_public", "use_env", "use_filter"):
        if type(value.get(name)) is not bool:
            raise ValueError("Invalid input setting: " + name)
    timeout = value.get("timeout_minutes")
    if type(timeout) is not int or timeout not in TIMEOUT_CHOICES:
        raise ValueError("Choose an MSA waiting limit of 10, 20, 30 or 60 minutes.")
    result = {name: value[name] for name in DEFAULT_SETTINGS}
    if result["mode"] != "Paste sequence":
        result["sequence"] = ""
    if result["mode"] == "Upload A3M":
        result["allow_public"] = False
    elif not result["allow_public"]:
        raise ValueError("Confirm public MSA search, or upload an existing A3M for a confidential sequence.")
    return result


def input_form(values=None, *, token=None):
    """Return safe HTML and its Promise expression; MSA options start collapsed."""
    initial = dict(DEFAULT_SETTINGS)
    if values:
        initial.update({key: value for key, value in values.items() if key in initial})
    # Ask for confirmation again when a cell is rerun, even if other fields persist.
    initial["allow_public"] = False
    if initial["mode"] not in INPUT_MODES:
        initial["mode"] = "Paste sequence"
    if not isinstance(initial["sequence"], str) or len(initial["sequence"].encode()) > 2 * 1024**2:
        initial["sequence"] = ""
    if initial["timeout_minutes"] not in TIMEOUT_CHOICES:
        initial["timeout_minutes"] = 20
    token = token or "trflow_input_" + uuid.uuid4().hex
    if not isinstance(token, str) or not re.fullmatch(r"trflow_input_[a-zA-Z0-9_]+", token):
        raise ValueError("Invalid form identifier.")
    options = "".join('<option' + (' selected' if mode == initial["mode"] else '') + '>'
                      + mode + '</option>' for mode in INPUT_MODES)
    timeouts = "".join('<option value="' + str(value) + '"'
                       + (' selected' if value == initial["timeout_minutes"] else '')
                       + '>' + str(value) + ' minutes</option>' for value in TIMEOUT_CHOICES)
    checked = lambda enabled: " checked" if enabled else ""
    markup = r'''<form id="TOKEN" class="trflow-input" style="font:14px/1.5 system-ui,sans-serif;color:#17212f;max-width:950px;border:1px solid #cbdde2;border-radius:10px;padding:18px;box-sizing:border-box">
<style>
#TOKEN [hidden]{display:none!important}
#TOKEN label{display:block;margin:10px 0 5px}
#TOKEN select,#TOKEN textarea{font:inherit;color:inherit;border:1px solid #9db4bd;border-radius:5px;padding:7px;box-sizing:border-box;max-width:100%;background:white}
#TOKEN textarea{width:100%;font-family:ui-monospace,monospace;resize:vertical}
#TOKEN details{margin:14px 0;padding:10px 12px;border:1px solid #dce5e8;border-radius:6px;background:#f7fafb}
#TOKEN summary{cursor:pointer;font-weight:600}
#TOKEN .tf-hint{font-size:12px;color:#4b6370;margin:5px 0}
#TOKEN button{font:inherit;background:#078DAF;color:white;border:0;border-radius:6px;padding:10px 16px;cursor:pointer;margin-top:10px}
#TOKEN button:disabled{opacity:.65;cursor:default}
</style>
<strong>Prepare one protein input</strong>
<label>Input mode <select data-role="mode" aria-label="Input mode">OPTIONS</select></label>
<div data-role="sequence-area"SEQUENCE_HIDDEN><label>SEQUENCE <textarea data-role="sequence" aria-label="SEQUENCE" rows="5" placeholder="Paste one protein sequence" spellcheck="false">SEQUENCE_VALUE</textarea></label></div>
<p data-role="upload-hint" class="tf-hint"UPLOAD_HIDDEN>Upload mode: click Continue below to show the file picker, then choose one file.</p>
<div data-role="msa-area"MSA_HIDDEN>
<label><input data-role="consent" type="checkbox"> I agree to send this sequence to the public ColabFold MSA service.</label>
<p class="tf-hint">For confidential sequences, use Upload A3M. Searches run serially; completed MSAs are cached in this runtime.</p>
<details data-role="advanced"><summary>Advanced MSA options · optional</summary>
<label>Search databases <select data-role="database" aria-label="Search databases"><option value="env"ENV_SELECTED>UniRef + environmental (default)</option><option value="uniref"UNIREF_SELECTED>UniRef only</option></select></label>
<label><input data-role="filter" type="checkbox"FILTER_CHECKED> Filter MSA hits (recommended)</label>
<label>Waiting time limit <select data-role="timeout" aria-label="Waiting time limit">TIMEOUT_OPTIONS</select></label>
<p class="tf-hint">Defaults match the standard ColabFold MSA search. The waiting limit does not change the alignment; a saved request can be resumed.</p>
</details>
</div>
<button type="submit" data-role="submit">Prepare input &amp; generate MSA</button>
<p data-role="status" role="status" aria-live="polite" class="tf-hint">Choose an input mode, then continue. File selection appears after you run this cell.</p>
</form>'''
    replacements = {"TOKEN": token, "OPTIONS": options,
                    "SEQUENCE_VALUE": html.escape(initial["sequence"]),
                    "SEQUENCE_HIDDEN": "" if initial["mode"] == "Paste sequence" else " hidden",
                    "UPLOAD_HIDDEN": " hidden" if initial["mode"] == "Paste sequence" else "",
                    "MSA_HIDDEN": " hidden" if initial["mode"] == "Upload A3M" else "",
                    "ENV_SELECTED": " selected" if initial["use_env"] else "",
                    "UNIREF_SELECTED": "" if initial["use_env"] else " selected",
                    "FILTER_CHECKED": checked(initial["use_filter"]), "TIMEOUT_OPTIONS": timeouts}
    markup = re.sub(r"\b(?:TOKEN|OPTIONS|SEQUENCE_VALUE|SEQUENCE_HIDDEN|UPLOAD_HIDDEN|MSA_HIDDEN|ENV_SELECTED|UNIREF_SELECTED|FILTER_CHECKED|TIMEOUT_OPTIONS)\b",
                    lambda match: replacements[match.group()], markup)
    expression = r'''(() => {
  let root = document.getElementById(TOKEN_JSON);
  if (!root) {
    const mount = document.createElement('div');
    mount.innerHTML = MARKUP_JSON;
    document.body.appendChild(mount);
    root = document.getElementById(TOKEN_JSON);
  }
  if (!root) throw new Error('The input panel is missing. Rerun Step 2.');
  const el = role => root.querySelector('[data-role="'+role+'"]');
  const mode = el('mode'), sequence = el('sequence'), consent = el('consent');
  function resize() {
    requestAnimationFrame(() => {
      if (globalThis.google?.colab?.output?.setIframeHeight) {
        google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
      }
    });
  }
  function updateMode() {
    const paste = mode.value === 'Paste sequence', search = mode.value !== 'Upload A3M';
    el('sequence-area').hidden = !paste;
    sequence.disabled = !paste; sequence.required = paste;
    el('upload-hint').hidden = paste;
    el('msa-area').hidden = !search;
    consent.disabled = !search; consent.required = search;
    el('submit').textContent = paste ? 'Prepare input & generate MSA' : (search ? 'Continue to FASTA upload' : 'Continue to A3M upload');
    el('status').textContent = paste ? 'Paste your sequence and confirm public MSA search.' : 'Click Continue to show the file picker below.';
    resize();
  }
  el('advanced').addEventListener('toggle', resize);
  mode.addEventListener('change', updateMode);
  updateMode();
  return new Promise(resolve => {
    root.addEventListener('submit', event => {
      event.preventDefault();
      if (!root.reportValidity()) return;
      const result = {mode:mode.value, sequence:mode.value === 'Paste sequence' ? sequence.value : '',
                      allow_public:mode.value !== 'Upload A3M' && consent.checked,
                      use_env:el('database').value === 'env', use_filter:el('filter').checked,
                      timeout_minutes:Number(el('timeout').value)};
      root.querySelectorAll('input,textarea,select,button').forEach(control => { control.disabled = true; });
      el('status').textContent = 'Settings accepted. The file picker or MSA progress will appear below.';
      resolve(result);
    });
  });
})()'''
    expression = re.sub(r"\b(?:TOKEN_JSON|MARKUP_JSON)\b",
                        lambda match: json.dumps(token if match.group() == "TOKEN_JSON" else markup), expression)
    return markup, expression
