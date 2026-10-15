"""Single-chain Colab inputs and a bounded, serial ColabFold MSA client.

Protocol reference: sokrypton/ColabFold, colabfold/colabfold.py, run_mmseqs2.
Only MSA search is requested, never AlphaFold prediction or template downloads.
Sequences are sent to the public service only after explicit notebook consent.
No third-party Python dependencies, Drive access or model deserialization.
"""
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
import time
from urllib import error, parse, request as urllib_request


API_URL = "https://api.colabfold.com"
API_MODES = {(True, True): "env", (False, True): "all",
             (True, False): "env-nofilter", (False, False): "nofilter"}
USER_AGENT = "trFlow-Colab/0.1 (https://github.com/YangLab-SDU/trFlow)"
AA = frozenset("ACDEFGHIKLMNPQRSTVWYX")
MAX_SEQUENCE = 10000  # Input guard, not a GPU-memory or prediction guarantee.
MAX_INPUT_BYTES = 2 * 1024**2
MAX_A3M_BYTES = 128 * 1024**2
MAX_ARCHIVE_BYTES = 64 * 1024**2
MAX_EXPANDED_BYTES = 256 * 1024**2
MAX_RECORDS = 100000
MAX_WAIT_SECONDS = 1200
POLL_SECONDS = 8
MAX_REQUEST_RETRIES = 3
MAX_SUBMISSIONS = 4
RESULT_MEMBERS = ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m")


class MSAError(ValueError):
    """The input, service response or cached alignment cannot be used safely."""


def normalize_sequence(text):
    """Accept plain sequence or exactly one FASTA record; never join proteins."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_INPUT_BYTES:
        raise MSAError("Provide one protein sequence or a FASTA file smaller than 2 MiB.")
    lines = [line.strip() for line in text.lstrip("\ufeff").splitlines() if line.strip()]
    if not lines:
        raise MSAError("Enter a protein sequence or upload one non-empty FASTA file.")
    if lines[0].startswith(">"):
        if not lines[0][1:].strip():
            raise MSAError("The FASTA header is empty.")
        lines = lines[1:]
    if any(line.startswith(">") for line in lines):
        raise MSAError("Use exactly one FASTA record; multiple proteins/complexes are not supported.")
    sequence = "".join("".join(line.split()) for line in lines)
    if not sequence.isascii():
        raise MSAError("Use ASCII amino-acid letters, not Unicode lookalikes.")
    sequence = sequence.upper()
    if not sequence or len(sequence) > MAX_SEQUENCE or set(sequence) - AA:
        raise MSAError("Use 1–10000 residues: the 20 standard amino acids or X. Gaps, stops, chain separators and residue numbers are not accepted.")
    return sequence


def parse_a3m(text, expected_query=None):
    """Validate the same single-chain query/aligned-column rules as trFlow."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_A3M_BYTES:
        raise MSAError("The A3M exceeds the 128 MiB input limit.")
    records, header, parts = [], None, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts)))
            if len(records) >= MAX_RECORDS:
                raise MSAError("The A3M has too many records.")
            header, parts = line[1:].strip(), []
            # MMseqs2 headers contain tab-separated alignment metadata. Tabs
            # are valid FASTA whitespace, not residue characters or code.
            if not header or len(header) > 4096 or any(ord(c) < 32 and c != '\t' for c in header):
                raise MSAError("The A3M contains an invalid header.")
        else:
            if header is None:
                raise MSAError("A single-chain A3M must start with >query, not multimer metadata.")
            parts.append(line)
    if header is not None:
        records.append((header, "".join(parts)))
    if not records:
        raise MSAError("The A3M contains no sequence records.")
    query = records[0][1]
    if not query or len(query) > MAX_SEQUENCE or set(query) - AA:
        raise MSAError("The first A3M record must be the uppercase, ungapped target sequence.")
    if expected_query is not None and query != expected_query:
        raise MSAError("The returned MSA query does not match the input protein sequence.")
    for _, sequence in records:
        if not sequence or set(sequence) - (AA | set("abcdefghijklmnopqrstuvwxyz") | {"-"}):
            raise MSAError("An A3M record contains unsupported residues or gaps.")
        aligned = "".join(c for c in sequence if not c.islower())
        if len(aligned) != len(query):
            raise MSAError("A3M aligned columns do not match the target sequence length.")
    return query, records


def _safe_directory(path):
    path = Path(path).expanduser().absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            raise MSAError("The MSA cache must not contain linked directories.")
    if path.exists() and not path.is_dir():
        raise MSAError("The MSA cache path is not a directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_file(path, limit):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise MSAError("Unsafe MSA-cache file: " + path.name)
    if not path.exists():
        return None
    if path.stat().st_size > limit:
        raise MSAError("An MSA-cache file exceeds its size limit: " + path.name)
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise MSAError("An MSA-cache file exceeds its size limit: " + path.name)
    return data


def _atomic_write(path, data):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise MSAError("Unsafe MSA-cache file: " + path.name)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".msa-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path, record):
    _atomic_write(path, (json.dumps(record, indent=2) + "\n").encode())


class _NoRedirect(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a submitted protein sequence to a redirected endpoint.
        return None


def _http_request(method, route, *, fields=None, limit, timeout):
    if not re.fullmatch(r"(?:ticket/msa|ticket/[A-Za-z0-9_-]{1,160}|result/download/[A-Za-z0-9_-]{1,160})", route):
        raise MSAError("Unexpected MSA service route.")
    body = None if fields is None else parse.urlencode(fields).encode()
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if body is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib_request.Request(API_URL + "/" + route, data=body, method=method, headers=headers)
    with urllib_request.build_opener(_NoRedirect()).open(req, timeout=timeout) as response:
        if response.status != 200 or response.geturl() != req.full_url:
            raise MSAError("The MSA service returned an unexpected HTTP response.")
        announced = response.headers.get("Content-Length")
        if announced is not None and (not announced.isdigit() or int(announced) > limit):
            raise MSAError("The MSA response exceeds the supported download size.")
        chunks, length = [], 0
        started = time.monotonic()
        while True:
            if time.monotonic() - started > timeout:
                raise MSAError("The MSA response download timed out.")
            block = response.read(min(1024**2, limit + 1 - length))
            if not block:
                break
            chunks.append(block)
            length += len(block)
            if length > limit:
                raise MSAError("The MSA response exceeds the supported download size.")
        if announced is not None and length != int(announced):
            raise MSAError("The MSA response download was incomplete.")
        return b"".join(chunks)


def _alignment_from_archive(data, sequence, check_time, *, use_env=True):
    """Read only reviewed A3M members; never extract service-controlled paths."""
    if not isinstance(data, bytes) or not data or len(data) > MAX_ARCHIVE_BYTES:
        raise MSAError("The MSA archive is empty or too large.")
    required_members = RESULT_MEMBERS if use_env else RESULT_MEMBERS[:1]
    members, total, count = {}, 0, 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r|gz") as archive:
            for member in archive:
                check_time()
                count += 1
                name = member.name
                path = PurePosixPath(name)
                if (count > 128 or path.is_absolute() or ".." in path.parts
                        or "\\" in name or ":" in name or not (member.isfile() or member.isdir())):
                    raise MSAError("The MSA server returned an unsafe archive.")
                total += member.size
                if member.size < 0 or total > MAX_EXPANDED_BYTES:
                    raise MSAError("The expanded MSA archive is too large.")
                if name not in required_members:
                    continue
                if not member.isfile() or name in members or member.size > MAX_A3M_BYTES:
                    raise MSAError("The MSA archive has invalid or duplicate alignment files.")
                stream = archive.extractfile(member)
                if stream is None:
                    raise MSAError("The MSA archive is missing alignment bytes.")
                with stream:
                    content = stream.read(MAX_A3M_BYTES + 1)
                if len(content) != member.size:
                    raise MSAError("The MSA archive contains an incomplete alignment.")
                # ColabFold uses NUL separators between query entries. Only one
                # query is submitted here; all records are still checked below.
                members[name] = content.decode("utf-8").replace("\x00", "")
    except (tarfile.TarError, OSError, UnicodeError, EOFError) as archive_error:
        raise MSAError("The MSA server did not return a readable alignment archive.") from archive_error
    if set(members) != set(required_members):
        raise MSAError("The MSA archive does not contain the alignments for the selected databases.")
    output, seen, count = [">query\n" + sequence + "\n"], {sequence}, 1
    for name in required_members:
        check_time()
        _, records = parse_a3m(members[name], expected_query=sequence)
        for header, aligned in records:
            if aligned in seen:
                continue
            seen.add(aligned)
            count += 1
            if count > MAX_RECORDS:
                raise MSAError("The combined MSA has too many records.")
            output.append(">" + header + "\n" + aligned + "\n")
    result = "".join(output)
    parse_a3m(result, expected_query=sequence)
    return result.encode("utf-8"), count


def ensure_msa(sequence, cache_dir, *, allow_public=False, progress=None,
               use_env=True, use_filter=True, timeout_minutes=None,
               requester=None, sleep=None, clock=None):
    """Reuse a verified runtime A3M or request one serial public MMseqs2 search.

    Cold requests require explicit consent. Pending tickets survive cancellation
    and are polled on reruns rather than resubmitted. No single-sequence fallback
    is manufactured when the service fails. Existing A3M upload remains available.
    """
    sequence = normalize_sequence(sequence)
    if type(use_env) is not bool or type(use_filter) is not bool:
        raise MSAError("MSA database and filtering settings must be boolean values.")
    if timeout_minutes is not None and (type(timeout_minutes) is not int or not 1 <= timeout_minutes <= 60):
        raise MSAError("The MSA waiting limit must be a whole number from 1 to 60 minutes.")
    mode = API_MODES[use_env, use_filter]
    max_wait = MAX_WAIT_SECONDS if timeout_minutes is None else timeout_minutes * 60
    requester = requester or _http_request
    sleep, clock = sleep or time.sleep, clock or time.monotonic
    started = clock()
    query_hash = hashlib.sha256(sequence.encode()).hexdigest()
    identity = {"schema_version": 1, "query_sha256": query_hash,
                "service": API_URL, "mode": mode}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = _safe_directory(_safe_directory(cache_dir) / key)
    alignment, receipt, ticket_path = (directory / name for name in ("alignment.a3m", "result.json", "ticket.json"))

    def note(message):
        if progress is not None:
            progress(message)

    def check_time():
        if clock() - started >= max_wait:
            raise MSAError("MSA search timed out. Rerun Step 2 to resume, or upload an existing A3M.")

    def read_json(path):
        data = _read_file(path, 1024**2)
        if data is None:
            return {}
        try:
            value = json.loads(data)
        except (ValueError, UnicodeError):
            return {}
        return value if isinstance(value, dict) else {}

    saved = read_json(receipt)
    existing = _read_file(alignment, MAX_A3M_BYTES)
    if existing is not None and all(saved.get(k) == v for k, v in identity.items()):
        if saved.get("sha256") == hashlib.sha256(existing).hexdigest():
            try:
                parse_a3m(existing.decode("utf-8"), expected_query=sequence)
            except (MSAError, UnicodeError):
                pass
            else:
                note("Using the saved, verified MSA")
                return alignment
    # Do not contact the public service, even to poll an old ticket, without consent.
    if allow_public is not True:
        raise MSAError("Enable ALLOW_PUBLIC_MSA to send this sequence to the public ColabFold service, or upload an existing A3M instead.")

    def call(method, route, fields=None, archive=False):
        for attempt in range(MAX_REQUEST_RETRIES):
            check_time()
            try:
                data = requester(method, route, fields=fields,
                                 limit=MAX_ARCHIVE_BYTES if archive else 1024**2,
                                 timeout=min(60, max(1, max_wait - (clock() - started))))
                if not isinstance(data, bytes) or len(data) > (MAX_ARCHIVE_BYTES if archive else 1024**2):
                    raise MSAError("The MSA response exceeds the supported size.")
                if archive:
                    return data
                result = json.loads(data)
                if not isinstance(result, dict) or not isinstance(result.get("status"), str):
                    raise MSAError("The MSA service returned an invalid job status.")
                return result
            except error.HTTPError as http_error:
                if http_error.code not in (429, 500, 502, 503, 504):
                    raise MSAError("The public MSA service rejected the request. Retry later or upload an existing A3M.") from http_error
            except (error.URLError, TimeoutError, ConnectionError, OSError):
                pass
            except (ValueError, UnicodeError) as response_error:
                raise MSAError("The MSA service returned an invalid response.") from response_error
            if attempt + 1 < MAX_REQUEST_RETRIES:
                note("MSA service temporarily unavailable; retrying serially")
                sleep(5 * (attempt + 1))
        raise MSAError("Could not reach the public MSA service. Rerun Step 2 or upload an existing A3M.")

    job = read_json(ticket_path)
    ticket = job.get("id")
    if not (all(job.get(k) == v for k, v in identity.items())
            and isinstance(ticket, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,160}", ticket)
            and job.get("state") in ("pending", "complete")):
        ticket = None
    if ticket is None:
        for attempt in range(MAX_SUBMISSIONS):
            note("Submitting one protein to the public ColabFold MSA service")
            result = call("POST", "ticket/msa", fields={"q": ">101\n" + sequence + "\n", "mode": mode})
            status = result["status"]
            if status in ("UNKNOWN", "RATELIMIT"):
                if attempt + 1 == MAX_SUBMISSIONS:
                    raise MSAError("The public MSA service is busy or rate-limited. Retry later or upload an existing A3M.")
                note("MSA service busy; waiting before another serial submission")
                sleep(10 * (attempt + 1))
                continue
            if status not in ("PENDING", "RUNNING", "COMPLETE"):
                raise MSAError("MSA search is unavailable (" + status[:40] + "). Retry later or upload an existing A3M.")
            ticket = result.get("id")
            if not isinstance(ticket, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", ticket):
                raise MSAError("The MSA service returned an invalid ticket identifier.")
            _write_json(ticket_path, dict(identity, id=ticket, state="pending"))
            break
    else:
        note("Resuming the saved MSA search request")
    while True:
        check_time()
        status = call("GET", "ticket/" + ticket)["status"]
        if status == "COMPLETE":
            break
        if status not in ("UNKNOWN", "PENDING", "RUNNING", "RATELIMIT"):
            _write_json(ticket_path, dict(identity, id=ticket, state="failed"))
            raise MSAError("MSA search did not finish (" + status[:40] + "). Retry later or upload an existing A3M.")
        note("Waiting for MSA search · " + status.lower())
        sleep(POLL_SECONDS if status != "RATELIMIT" else 20)
    _write_json(ticket_path, dict(identity, id=ticket, state="complete"))
    note("Downloading and validating the generated MSA")
    data, count = _alignment_from_archive(call("GET", "result/download/" + ticket, archive=True), sequence, check_time, use_env=use_env)
    check_time()
    _atomic_write(alignment, data)
    _write_json(receipt, dict(identity, sha256=hashlib.sha256(data).hexdigest(),
                              bytes=len(data), records=count, query_length=len(sequence)))
    note("MSA ready · " + str(count) + " sequences")
    return alignment
