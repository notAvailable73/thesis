# Kaggle runbook — TMLR experiments (one operator, 4 Kaggle accounts)

You launch sessions; the AI prepares shards and reads the results. Everything runs from **one notebook**, [`notebooks/tmlr_runner.ipynb`](../notebooks/tmlr_runner.ipynb). For each new shard you only edit its first code cell (the control cell).

Decisions on record (plan A11): **D1** FGVC-Aircraft · **D4** Hugging Face central storage: yes.

---

## Part 1 — One-time setup

### 1.1 Hugging Face (central result storage) — about 10 minutes

Every finished shard uploads its zip here automatically, from any of the 4 accounts. You never have to download Kaggle outputs one by one, and results always exist in at least two places (Kaggle output plus HF).

1. **Create an account** at <https://huggingface.co/join> (free) and confirm your email.
2. **Create a private dataset repo.** Go to <https://huggingface.co/new-dataset>, set **Name** to `bpeft-tmlr-runs`, **Visibility** to **Private**, and click **Create dataset**. Your repo id is `<your-hf-username>/bpeft-tmlr-runs`; write it down.
3. **Create a write token.** In your avatar menu go to **Settings → Access Tokens → Create new token**.
   - Simplest: **Token type: Write**. Name it `kaggle-tmlr`.
   - Tighter alternative: **Fine-grained**, with *write* access to `bpeft-tmlr-runs` only.
   - Copy the token (starts with `hf_…`). It is shown only once.
4. **Local machine** (to download results later), one-time:
   ```bash
   uv tool install huggingface_hub     # or: pip install -U huggingface_hub
   huggingface-cli login               # paste the same token (newer versions also accept: hf auth login)
   ```
5. The token goes into each Kaggle account in step 1.2.5. Never paste it into a notebook cell or into Git.

### 1.2 Each of the 4 Kaggle accounts (acct1 … acct4)

1. **Phone verification.** Go to *Settings → Phone verification*. It is required for GPU and Internet.
2. **Check your GPU quota.** It is shown under *Settings* or on the right panel when a GPU session is on. Write down the weekly hours for each account.
3. **Data access.** The notebook needs CIFAR-100 (`cifar-100-python`), SVHN `test_32x32.mat`, `tiny-imagenet-200` and the MiniImageNet pickles.
   - If the Kaggle dataset `beft-thesis-data` is on one of your accounts, open it, go to *Settings → Sharing*, and add your other three usernames.
   - If it was on a former teammate's account, re-create it on your main account. Use *Datasets → New Dataset*, private, upload the same four sources, then share it with your other three accounts. The notebook finds the files at any depth, so the dataset name does not matter.
4. **Import the notebook.** *Create → New Notebook*, then *File → Import Notebook*, and upload `notebooks/tmlr_runner.ipynb` from your repo checkout. Rename it, for example to `bpeft-tmlr-runner`.
5. **Notebook settings** (right sidebar):
   - *Session options → Accelerator*: **GPU T4 x2**.
   - *Session options → Internet*: **On**.
   - *Add Input → Datasets*: your data dataset from step 3.
   - *Add-ons → Secrets → Add a new secret*: label **`HF_TOKEN`**, value = your `hf_…` token. Make sure it is **attached to this notebook** (toggle it on).
6. In the control cell, set `HF_REPO = '<your-hf-username>/bpeft-tmlr-runs'` once.

---

## Part 2 — Launching a shard (repeat for every shard)

1. Open the account's runner notebook and edit the **control cell**:
   - `SHARD_FILE`: for example `configs/tmlr/shards_w0.json`.
   - `SHARD_ID`: for example `W0-01`, from the schedule table below.
   - `GIT_SHA`: the 40-character commit the AI gives you. It is identical for every shard of a wave.
2. **Save Version** (top right), choose **Save & Run All (Commit)**, then **Save**.
3. **Close the tab.** The run continues in the background, up to 12 hours.
4. **Monitor** from the notebook page → *Versions* → the running version → *Logs*. Every 2 minutes it prints `done X/Y` and the last line of each lane.
5. **When it ends:**
   - **"Complete":** the zip is on HF and in the version's *Output*.
   - **"Error":** everything that finished was still packed and uploaded. Tell the AI; missing runs go into a retry shard.

**Rules**
- One GPU session per account at a time. Start an account's next shard only after its current one ends.
- **Never edit code inside Kaggle.** Code changes go through Git and a new `GIT_SHA`.
- Never lower `NUM_EPISODES` for a real run.
- **Resuming a shard that died:** before *Save Version*, use *Add Input → Your Work → this notebook* and select the previous version's output. Finished runs are restored and skipped.

---

## Part 3 — After each wave: collect

```bash
scripts/tmlr_pull.sh hf <your-hf-username>/bpeft-tmlr-runs
uv run --frozen python scripts/tmlr_collect.py --incoming incoming/
```

Send the AI the printed report, or just say "collected". It checks coverage and the reproduction guards, then builds the next wave's shards from the measured run times.

---

## Part 4 — Schedule

### Wave 0 — validation (all 4 accounts at once, about 30–60 min each including setup)

| Account | Shard | Runs |
|---|---|---|
| acct1 | `W0-01` | MNv3 lora_ss L (regression anchor), MNv2 btl_ss H, R34 btl_ms H |
| acct2 | `W0-02` | grid cell CIFAR 1-shot (regression anchor), R18 btl_ms H (regression anchor) |
| acct3 | `W0-03` | R18 btl_ss H (stride-2 site), R18 lora_ms H |
| acct4 | `W0-04` | E5-val (VAL-only path), MNv2 lora_ss unmatched |

**Gate after Wave 0** (the AI checks it):
- 9/9 done.
- All 3 regression anchors `exact` or `within_tol`.
- Parameter counts match.
- GPU is T4.
- The new score keys are present.

**Also write down for each account:** the quota before and after (does T4 x2 count once or double?) and the wall time per run.

### Wave 1 — E1 + E2 + E5-val (≈348 runs)

The AI fills this table after Wave 0, from the measured timings.

### Wave 2 — E3a + E3b + E5 + E4 (≈253 runs; Aircraft code is built during Wave 1)

The AI fills this table after Wave 1.
