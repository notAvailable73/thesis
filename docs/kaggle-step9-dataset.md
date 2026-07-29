# Kaggle dataset layout for `notebooks/step9-mini.ipynb`

Step 9's first Kaggle attempt (commit `6c42070`, 2026-07-28) trained and
evaluated all 10 MiniImageNet configs successfully, but its test cell failed:
nothing was attached, so `get_cifar_fs` fell through to a runtime download and
`cs.toronto.edu` served ~150-200 KB/s and timed out repeatedly. **CIFAR-100 is
the reason this document exists.** Everything else in the list below is a
time saver, not a fix.

The notebook now expects **one** attached dataset. Its single top-level folder
must be named exactly `bpeft-data`.

## 1. The dataset

| field | value |
|---|---|
| Kaggle dataset title | `bpeft-thesis-data` |
| Top-level folder inside it | **`bpeft-data`** ← the name the notebook searches for |
| Visibility | Private is fine |
| Size | 10 files, ~2.3 GB |

```
bpeft-data/
├── cifar-100-python/
│   ├── meta                                  (   0.0 MB)
│   ├── train                                 ( 155.2 MB)
│   └── test                                  (  31.0 MB)
├── svhn/
│   └── test_32x32.mat                        (  64.3 MB)
├── miniimagenet/
│   ├── mini-imagenet-cache-train.pkl         (1145.5 MB)
│   ├── mini-imagenet-cache-validation.pkl    ( 292.7 MB)
│   └── mini-imagenet-cache-test.pkl          ( 353.6 MB)
├── tinyimagenet/
│   └── tiny-imagenet-200.zip.dat             ( 248.1 MB)
└── splits/
    ├── cifar_fs_split.json                   (   2 KB)
    └── mini_imagenet_split.json              (   2 KB)
```

A ready-to-upload copy of exactly this tree already exists on the dev machine
at `../thesis-kaggle-upload/bpeft-data/` (hardlinks into `data/`, so it costs
no extra disk). Upload that folder; do not rearrange it.

### Why each choice

* **`cifar-100-python/` extracted, not the tarball.** The notebook symlinks the
  directory straight into `data/cifar-100-python`, which is where
  `torchvision.datasets.CIFAR100(root='data', download=False)` looks. Its md5
  integrity check passes through the symlink (verified). Uploading
  `cifar-100-python.tar.gz` also works — Kaggle auto-extracts it to the same
  directory — but then the archive is on the mount as well, for nothing.
  `file.txt~` from the original tarball is deliberately **not** included;
  nothing reads it.
* **`tiny-imagenet-200.zip.dat`, not `.zip` and not extracted.** Kaggle
  auto-extracts uploaded `.zip` archives, and an extracted TinyImageNet is
  ~120k files: that makes every directory scan of `/kaggle/input` expensive and
  the dataset unpleasant to manage. The `.dat` suffix defeats the
  auto-extraction; Section 2 links it back to `data/tiny-imagenet-200.zip`,
  which is the name `ensure_archive()` expects, and
  `src/datasets/tinyimagenet_ood.py` reads the 500 images it needs straight out
  of the archive without extracting (repo rule: never extract a many-file
  archive). An extracted `tiny-imagenet-200/` tree is still detected and linked
  if you prefer to attach one — the resulting OOD pool is the identical 500
  images either way (both paths sort a constant-prefixed name list before
  `rng.sample`).
* **The 3 Zenodo pkls, not the `.npy` caches.** They are the loader's
  documented primary source and are MD5-checked against
  `_ZENODO_FILES`. Section 5 decodes them into `data/mini_imagenet_84_*.npy`
  (~1.3 GB) once per session.
* **`splits/*.json` are the offline fallback** for
  `scripts/build_cifar_fs_split.py` / `build_mini_imagenet_split.py`, which
  fetch from `raw.githubusercontent.com`. Section 2 **copies** these (not
  symlinks) because those scripts rewrite them and `/kaggle/input` is read-only.
  Both must carry a canonical `_status`; Section 3 asserts they are not the
  synthetic fallback.

## 2. Uploading

Browser: *Datasets → New Dataset → Upload folder*, pick the `bpeft-data`
folder, title it `bpeft-thesis-data`, Create. Or:

```bash
cd ../thesis-kaggle-upload
kaggle datasets init -p .
# edit dataset-metadata.json: "title": "bpeft-thesis-data", "id": "<user>/bpeft-thesis-data"
kaggle datasets create -p . --dir-mode zip
```

Verify afterwards that the file browser shows `bpeft-data/` as the only
top-level entry and that `tiny-imagenet-200.zip.dat` is still a single file
(not an unpacked tree).

## 3. Running

1. New notebook → **Settings: Accelerator = GPU T4, Internet = ON**.
2. `+ Add Input` → your `bpeft-thesis-data` dataset.
3. Upload `notebooks/step9-mini.ipynb` and run top to bottom. No cell needs
   editing.

Section 2 finds `bpeft-data` by name at any depth from `/kaggle/input/bpeft-data`
down to four levels in, so it works on both mount layouts Kaggle uses
(`/kaggle/input/<slug>/…` and `/kaggle/input/datasets/<owner>/<slug>/…`).
Section 2b prints every staged file's MD5 against torchvision's own
`CIFAR100.train_list` / `SVHN.split_list` constants and `_ZENODO_FILES`;
Section 2c then makes each real loader accept the layout with `download=False`.
If a piece is missing, it says `MISS` and that dataset downloads at runtime —
nothing hard-fails.

## 4. Disk / output notes

* `/kaggle/working` (20 GB): the run adds ~1.3 GB of `.npy` MiniImageNet caches
  plus ~40 MB of checkpoints. The 2.3 GB of attached data is symlinked, never
  copied.
* Those `.npy` caches live under `/kaggle/working/thesis/data/` and therefore
  count toward the notebook **Output** if you use *Save Version*. In an
  interactive session, prefer the download link Section 9 prints
  (`step9_mini_artifacts.zip`, ~50 MB) and skip committing a version.
* If the session restarts, re-run Sections 2, 3 and 5 before Section 7.
  Section 7 is resumable: a config whose result JSON already exists is skipped.
