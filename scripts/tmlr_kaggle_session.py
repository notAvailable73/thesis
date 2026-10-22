"""TMLR Kaggle session body: data staging, preflight, data preparation, restore,
the GPU lanes, packing/upload, verdict (plan Task 9).

Executed by notebooks/tmlr_runner.ipynb AFTER it has checked out GIT_SHA, via
    exec(open('scripts/tmlr_kaggle_session.py').read())
so it runs in the notebook's own kernel (output can't vanish) and reads the
control-cell variables (SHARD_FILE, SHARD_ID, GIT_SHA, LANES, MAX_MINUTES,
NUM_EPISODES, PERSIST_LOGITS, UPLOAD_TO_HF, HF_REPO) and REPO / OUT from it.
Keeping this in the repo means a fix reaches Kaggle by bumping GIT_SHA; the
uploaded notebook never has to change again.
"""
# ruff: noqa
# ---- 3. data: symlink attached datasets into data/ ---------------------------
from src.datasets.cifar_fs import _find_staged_cifar100_root
from src.datasets.svhn_ood import _find_staged_svhn_root
from src.datasets.tinyimagenet_ood import _find_extracted_tin_root
from src.datasets.mini_imagenet import _find_zenodo_pkls
LINKS = {}
if (root := _find_staged_cifar100_root('data')):
    LINKS['data/cifar-100-python'] = os.path.join(root, 'cifar-100-python')
if (root := _find_staged_svhn_root('data')):
    LINKS['data/svhn/test_32x32.mat'] = os.path.join(root, 'test_32x32.mat')
if (root := _find_extracted_tin_root('data')):
    LINKS['data/tiny-imagenet-200'] = root
for _split, _pkl in (_find_zenodo_pkls('data') or {}).items():
    LINKS[f'data/{_pkl.name}'] = str(_pkl)
try:                                   # FGVC-Aircraft cache (Wave 2 onwards)
    from src.datasets.fgvc_aircraft import _find_aircraft_cache
    if (p := _find_aircraft_cache('data')):
        LINKS[f'data/{p.name}'] = str(p)
except ImportError:
    pass
os.makedirs('data/svhn', exist_ok=True)
for link, target in LINKS.items():
    if os.path.exists(link) or os.path.islink(link):
        print('present  ', link); continue
    os.symlink(target, link); print('linked   ', link, '->', target)
if not LINKS:
    print('WARNING: nothing staged under /kaggle/input; datasets will try to download at runtime')
# frozen splits (idempotent: validated against the committed constants, never regenerated)
subprocess.run([sys.executable, 'scripts/build_mini_imagenet_split.py'], check=True)
subprocess.run([sys.executable, 'scripts/build_cifar_fs_split.py'], check=True)

# ---- 4. preflight (CPU): this shard's runs exist, budgets match, recipe controlled
import tmlr_design as T, tmlr_configs as C, tmlr_run_shard as RS
SHARDS = {sh['shard_id']: sh for sh in json.load(open(SHARD_FILE))['shards']}
SHARD = SHARDS[SHARD_ID]
DESIGN = {s.run_id: s for s in T.build_design()}
mine = [DESIGN[r] for lane in SHARD['lanes'] for r in lane]
assert len(SHARD['lanes']) <= LANES or all(not l for l in SHARD['lanes'][LANES:]), \
    f'shard has {len(SHARD["lanes"])} lanes but LANES={LANES}'
budget = C.assert_budgets(mine)
guard = C.control_guard(mine)
print(f'{len(mine)} runs | budgets {"OK" if budget["ok"] else "FAIL"} | control {"OK" if guard["ok"] else "FAIL"}')
for f in budget['failures'] + guard['failures']:
    print('  FAIL', f)
assert budget['ok'] and guard['ok'], 'refusing to spend GPU time'
for i, lane in enumerate(SHARD['lanes']):
    print(f'lane {i}: {len(lane)} runs, ~{SHARD["est_minutes"][i]:.0f} min estimated')
# Download/decode every dataset + OOD source ONCE, here, before the lanes start:
# two lanes preparing the same data at once corrupt each other's cache (W0-01 crash).
t_prep = time.time()
PREPARED = RS.prepare_data(mine)
print(f'data ready in {(time.time() - t_prep) / 60:.1f} min:', PREPARED)

# ---- 5. restore finished runs from an attached previous output ---------------
def _tmlr_out_dirs(root='/kaggle/input', max_depth=6):
    skip = {'images', 'tiny-imagenet-200', 'cifar-100-python', 'train', 'val', 'test'}
    frontier, hits = [Path(root)], []
    for _ in range(max_depth):
        nxt = []
        for d in frontier:
            try:
                for e in os.scandir(d):
                    if e.is_dir() and e.name not in skip:
                        (hits if e.name == SHARD_ID and 'tmlr_out' in e.path else nxt).append(Path(e.path))
            except OSError:
                pass
        frontier = nxt
    return hits
restored = 0
for src_root in _tmlr_out_dirs() if os.path.isdir('/kaggle/input') else []:
    for s in mine:
        src = src_root / s.experiment / s.run_id
        dst = OUT / s.experiment / s.run_id
        if (src / 'metrics.json').exists() and not (dst / 'metrics.json').exists():
            shutil.copytree(src, dst, dirs_exist_ok=True); restored += 1
print('restored', restored, 'finished run(s) from earlier sessions')

# ---- 6. run the lanes (one process per GPU), pack + upload no matter what ----
procs, logs = [], []
def _pack_and_upload():
    rep = RS.shard_report(SHARD, OUT, DESIGN)
    (OUT / 'shard_report.json').write_text(json.dumps(rep, indent=1, sort_keys=True))
    zpath = Path('/kaggle/working' if os.path.isdir('/kaggle/working') else REPO) / f'{SHARD_ID}.zip'
    lines = []
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():
                rel = p.relative_to(OUT)
                zf.write(p, rel)
                lines.append(f'{rel}  sha256={hashlib.sha256(p.read_bytes()).hexdigest()}')
        zf.writestr('MANIFEST.txt', '\n'.join(lines))
    print(f'packed {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)')
    if UPLOAD_TO_HF:
        try:
            from kaggle_secrets import UserSecretsClient
            from huggingface_hub import HfApi
            token = UserSecretsClient().get_secret('HF_TOKEN')
            HfApi(token=token).upload_file(path_or_fileobj=str(zpath), repo_type='dataset',
                path_in_repo=f'shards/{SHARD_ID}.zip', repo_id=HF_REPO,
                commit_message=f'{SHARD_ID} @ {GIT_SHA[:8]}: {len(rep["done"])}/{rep["expected"]} done')
            print('uploaded to', HF_REPO)
        except Exception as e:
            print('HF UPLOAD FAILED (the Kaggle output still has the zip):', repr(e))
    return rep

try:
    for i, lane in enumerate(SHARD['lanes'][:LANES]):
        if not lane:
            continue
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(i), 'GIT_SHA': GIT_SHA, 'OMP_NUM_THREADS': '2'}
        log = open(OUT / f'_lane{i}.log', 'a')
        cmd = [sys.executable, 'scripts/tmlr_run_shard.py', '--shard-file', SHARD_FILE, '--shard-id', SHARD_ID,
               '--lane', str(i), '--out-dir', str(OUT), '--max-minutes', str(MAX_MINUTES),
               '--num-episodes', str(NUM_EPISODES)] + ([] if PERSIST_LOGITS else ['--no-logits'])
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO))
        logs.append(OUT / f'_lane{i}.log')
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(120)
        done = len(list(OUT.rglob('metrics.json')))
        tails = []
        for lp in logs:
            lines = lp.read_text(errors='replace').strip().splitlines()
            tails.append(lines[-1][:110] if lines else '')
        print(f'[{(time.time() - t0) / 60:5.0f} min] done {done}/{len(mine)} | ' + ' | '.join(tails), flush=True)
    print('lane exit codes:', [p.returncode for p in procs])
finally:
    for p in procs:
        if p.poll() is None:
            p.kill()
    REPORT = _pack_and_upload()

# ---- 7. verdict: fail the version visibly if anything is missing -------------
print(json.dumps({k: v for k, v in REPORT.items() if k != 'done'}, indent=1))
bad_guards = {k: v for k, v in REPORT['guards'].items() if k not in ('exact', 'within_tol')}
if bad_guards:
    print('REGRESSION GUARD PROBLEM — stop the campaign and debug before more waves:', bad_guards)
assert not REPORT['missing'] and not bad_guards, \
    f'{len(REPORT["missing"])} run(s) missing/errored -> they go into a retry shard'
print('shard complete')
