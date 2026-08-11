# Step 9 — Testing the Renovation on a Different Neighborhood (MiniImageNet Dataset)

> Continues from [Step 8](step8.md); see [Step 5](step5.md) for the core vocabulary
> (backbone, adapter, few-shot learning, softmax vs. evidential head).

## 1. What we did

Every step so far tested the model on one specific set of images: **CIFAR-FS**, a
dataset built from small, low-resolution photos. Step 9 added a **second dataset**,
called **MiniImageNet** (100 categories, 600 images each), so that all the project's
findings could be checked against more than one source of photos, not just one
benchmark.

The plan was to re-run the same core comparisons already used in Steps 6 and 8 — both
backbones (ResNet-18 and MobileNetV3-Small), both placements (post_pool and the
Step 6-winning parallel), both heads (softmax and evidential) — but on MiniImageNet
instead of CIFAR-FS, plus a "linear-probe" reference run (see [Step 5](step5.md): the
backbone frozen, with no adapter at all) specifically added to measure a confound
explained below. That's 10 configurations, each tested over 600 episodes, matching the
pattern of every step before it.

**What happened while building it:** the code was written and validated as far as
possible without running it, then a separate session started running the real training
notebook on Kaggle (a cloud service with a free GPU) and hit two real bugs along the
way — a setup step that crashed when a dataset file wasn't already attached (even
though a working download fallback already existed for it), and a MiniImageNet download
that failed everywhere because the file host was blocking the specific browser
identification string the code was sending. Both were found and fixed during that same
session, and locked in with new automated tests so they can't silently reappear.

**What happened after the fixes:** all 10 configurations then ran to completion on a
Kaggle GPU in one sitting (about 2 hours 22 minutes end to end), and the results have
now been transcribed into the project's official results table. Step 9 is closed.

## 2. Why we did it

Steps 5–8 built up a set of findings — which adapter method works best, which placement
works best, how much a smaller backbone costs — using CIFAR-FS as the only source of
images. Any one of those findings could, in principle, be a quirk of that specific
dataset rather than something generally true. Testing a second, different dataset is the
standard way to check that.

MiniImageNet was chosen specifically because it introduces an interesting wrinkle worth
measuring on purpose: its 100 categories are drawn from ImageNet, the same enormous
photo collection the backbones were *originally* pretrained on. That means the backbone
may have already "seen" these categories (in a general sense) long before this project's
few-shot task even begins — unlike CIFAR-FS's categories, which are more clearly novel
to it. Prior research (cited in the underlying write-up) found that when a frozen
backbone already has strong prior exposure to the categories being tested, even the
simplest possible baseline — no adapter at all — becomes surprisingly competitive. The
linear-probe reference run mentioned above exists specifically to *measure* that effect
directly, rather than just assuming it — and, as Section 5 below shows, it did.

## 3. The analogy

Every renovation so far — the contractor comparisons in [Step 5](step5.md), the
placement study in [Step 6](step6.md), the security-system test in
[Step 7](step7.md), and the smaller studio apartment in [Step 8](step8.md) — happened
in the same neighborhood, using photos of the same kind of houses to judge the work.
Step 9 takes the exact same renovation blueprint and tests it on a **house in a
different neighborhood** — a different city, built from a different, larger catalog of
building plans, some of which the very same architects who designed our original house
had already worked from before.

Here's the catch worth measuring on purpose: since the new neighborhood's houses were
partly designed by architects who also designed our original house's "bones" (the
pretrained backbone), a renovation crew might do suspiciously well here even with almost
no renovation effort at all — because the house was already halfway suited to the new
tenants before anyone picked up a tool. So alongside the real renovation crews, we also
sent in a crew that does **absolutely nothing** to the new house, just to see how well
"doing nothing" scores on its own — a fair baseline for judging whether the real
renovation crews are actually adding value here, or just riding on a head start.

The inspection, once it finally happened, confirmed exactly that suspicion: houses in
the new neighborhood scored noticeably higher across the board — even the "do nothing"
crew scored unusually well — and the real renovation crews' edge over "doing nothing"
shrank to roughly a third to a half of what it had been back in the original
neighborhood. The renovation crews are still adding real value, just measurably less of
it here, because the house needed less help to begin with. And one more surprise showed
up only after the inspection: the smaller studio-apartment house from Step 8, which had
cost almost nothing in quality back in the original neighborhood, turned out to cost
noticeably more here — a gap nobody predicted from the blueprints alone, and one the
project is flagging rather than explaining away.

## 4. Mapping analogy to reality

| Analogy element | Real technical thing |
|---|---|
| A house in a different neighborhood, different city | **MiniImageNet** — a second, different image dataset |
| The new neighborhood's architects overlapping with our original house's architects | MiniImageNet's 100 categories come from ImageNet — the same huge collection the backbones were pretrained on |
| A crew that does nothing to the house, as a fair-baseline check | The **linear-probe reference run** — backbone frozen, no adapter at all |
| Houses scoring higher across the board, even the do-nothing one | MiniImageNet accuracy is 4–9 percentage points higher than the matched CIFAR-FS result for 9 of the 10 configurations |
| The real crews' edge over "doing nothing" shrinking | The adapter's accuracy boost over the linear-probe baseline shrinks from CIFAR-FS's ~+3.8–4.1pp to MiniImageNet's ~+1.4–1.9pp |
| The smaller studio apartment costing more here, unexpectedly | MobileNetV3-Small's accuracy cost vs. ResNet-18 balloons from 0.3–1.6pp (CIFAR-FS, Step 8) to 4.7–5.7pp (MiniImageNet) — unexplained, flagged for Step 10 |

## 5. Benefit / what it improved

All numbers below are copied directly from the project's results
(`phase5_dataset_table.json`; 600 test episodes per configuration):

- **The "pretrained-overlap" concern from Section 2 is now a measured number, not a
  guess.** The best adapter's accuracy edge over the do-nothing linear-probe baseline is
  CIFAR-FS: +3.8 to +4.1 percentage points, vs. MiniImageNet: +1.4 to +1.9 percentage
  points — roughly a third to a half the size. On MiniImageNet, the frozen backbone
  already "knew" enough about these categories that the trained adapter has noticeably
  less room to add value over doing nothing at all.
- **Overall accuracy is 4–9 percentage points higher on MiniImageNet than on CIFAR-FS
  for 9 of the 10 matched comparisons** — including the single best accuracy recorded
  anywhere in this project so far (96.0%, ResNet-18 + parallel placement + evidential
  head). One combination bucked the trend and actually scored slightly *lower* on
  MiniImageNet (MobileNetV3-Small + parallel placement + softmax head, by about 1.1
  percentage points) — reported honestly rather than smoothed over, since it's the one
  place the "MiniImageNet is easier" story doesn't hold.
- **The evidential head's advantage over plain softmax confidence survived the dataset
  swap completely** — across 16 test-set/configuration comparisons, it beat both
  probability-based confidence scores every single time, without exception. Against the
  third, non-probability-based "energy" score, though, the picture got noticeably worse
  than it was on CIFAR-FS: evidential now wins only 7 of those same 16 comparisons,
  extending a weakness Step 8 first spotted on the smaller backbone.
- **The calibration gap (how well the model's stated confidence matches reality) is the
  widest yet recorded.** On ResNet-18, the evidential head's calibration error is now
  55–57× worse than a properly corrected softmax score's — well beyond the previous
  worst gap (35×, from Step 8). This strengthens an earlier finding that the evidential
  head's calibration doesn't really respond to how easy or hard the underlying task is;
  it stays roughly the same difficulty-independent number, while the softmax side keeps
  getting better as the task gets easier.
- **A known issue reproduced on an entirely different dataset.** One specific
  configuration (MobileNetV3-Small + parallel placement + evidential head) had its final
  checkpoint saved too early in training back in Step 8, before an internal calibration
  dial had finished ramping up — and it had that step's worst calibration score as a
  result. The exact same thing happened again here, on MiniImageNet, to the exact same
  configuration. Seeing it twice on two different datasets makes this a real pattern
  worth fixing in Step 10, not a one-off fluke.
- The automated test suite ran cleanly except for 6 failures, all traced to the same
  harmless cause: this particular test run happened on a machine that already had a
  real dataset attached, which tripped up a handful of tests written to expect a
  completely empty environment. None of the 6 point to an actual bug in the new code.

## 6. How this fits the overall thesis

Step 9 extends two of the thesis's research questions (RQ2, about calibrated
confidence, and RQ3, about detecting unfamiliar images) so they now rest on more than
one dataset — a meaningfully stronger thesis claim than "true on CIFAR-FS" alone. It
also delivered on a direct, honest measurement of a real risk to the thesis's story:
whether a do-nothing baseline would score suspiciously close to the carefully engineered
adapters on this dataset. It does score closer here than on CIFAR-FS, and the project
reports that plainly rather than avoiding it — while also showing the adapters still add
real, measurable value on top of it. The unexpected finding that a small, "nearly free"
backbone from Step 8 turns out not to be nearly free on this second dataset is exactly
the kind of result a second dataset is supposed to surface, and it now sits as an open
question for Step 10 rather than a settled conclusion.

## 7. What's next, and why

According to `progress.txt`, the project's next step is **Step 10: a large grid of
runs** (roughly 120 combinations of adapters, placements, backbones, and datasets)
meant to properly re-tune settings for each backbone/dataset combination individually —
something every step from 5 through 9 explicitly deferred, since each of them reused one
fixed recipe (originally tuned only for ResNet-18 on CIFAR-FS) rather than tuning it
fresh each time. Step 9 raises the stakes on that deferred work in two concrete ways:
the early-checkpoint calibration issue now has two independent occurrences pointing at
it, and the surprising jump in the small backbone's cost on this new dataset is a real
open question that per-dataset tuning is specifically positioned to help answer.

---

# Addendum — Step 9 v2: the same experiments, run through the GPU relay

> ⚠️ **This is a second notebook for the exact same Step 9, not a new step.** Everything
> above (Sections 1–7) describes `notebooks/step9-mini.ipynb` (and the completed,
> saved-output copy of it, `step9-mini(1).ipynb`) — the plain, one-session, "run the
> whole thing inside Kaggle/Colab" path that is this project's default (see CLAUDE.md:
> the GPU relay is opt-in, brought in here only because this notebook names it).
> `notebooks/step9-mini-v2.ipynb` reruns the identical 10 configurations through the
> opt-in GPU relay system (`src/vgpu/`, `docs/gpu-relay-guide.md`) instead. As of this
> write-up, its own saved cell outputs still only show the relay pipeline validated
> through the start of data staging — see Section 1 below for exactly how far it got —
> and it is **not** what ended up producing Step 9's real results; see the update at the
> end of this section.

## 1. What we did

`step9-mini-v2.ipynb` runs the same ten configurations as the plain notebook (both
backbones × both placements × both heads, plus the ResNet-18 linear-probe reference) but
splits the work across two places instead of one: a **local control side** (this repo, on
whatever machine you run the notebook from) that holds the source code, the datasets, the
configs, the results, and the checkpoints; and a **remote GPU side** (a Kaggle Tesla T4,
borrowed through a shared pool called the "PixelPals GPU Relay") that only ever receives
the specific pieces of work that actually need a GPU — initializing a model, running one
epoch of training, evaluating one shard of test episodes, fitting the softmax temperature,
exporting a checkpoint. Everything else (choosing what to run next, keeping the results
table, building the plots, deciding what "done" means) stays local.

The notebook's own saved outputs (from a run on 2026-07-30) show it got this far before
stopping:

1. **Local gate passed in full.** The complete repository test suite ran locally and
   passed — **213 passed**, no failures — a stricter bar than the plain notebook's own
   run cleared on Kaggle (see the caveat below).
2. **Source and data were hashed for the relay, not just for git.** The notebook bundled
   105 source files and hashed the local `data/` folder's contents (**120,214 files,
   2.356 GiB** — including the CIFAR-100 tarball contents, `cifar_fs_split.json`, and the
   two MiniImageNet Zenodo pickle caches) into a manifest, so every byte sent to the
   remote worker is checked against a known-good hash before and after transfer.
3. **The verified relay probe passed.** The exact standalone probe from
   `docs/gpu-relay-guide.md` ran first and printed its literal `VERIFIED` line: it
   connected to the live control plane, picked a connected Kaggle T4 ("Tesla T4", 14.56
   GiB free), ran a real 4096×4096 CUDA matrix multiply on it (7.09 ms), and confirmed
   teardown — proving the whole path (auth → node → real GPU work → teardown) works
   before any real experiment data was risked on it.
4. **The relay connected and deployed its worker.** A temporary, checksum-pinned worker
   script was accepted by the Kaggle node (matching the SHA-256 printed locally), and the
   run started from a clean `resume state: idle` — nothing had been run on this remote
   worker yet.
5. **Resumable data staging started, then stopped mid-transfer.** The notebook
   deliberately interrupts itself once early on, on purpose, to prove the resume logic
   actually works (one source-code chunk, one small data-file chunk) — both of those
   planned interruptions succeeded. After that, it moved on to the two large MiniImageNet
   cache files (the 353.6 MB test cache and the 1,145.5 MB train cache) and was stopped by
   a `KeyboardInterrupt` partway through — this last stop does **not** look like one of the
   notebook's own planned tests; it reads as the run being manually interrupted while
   uploading the largest files in the manifest.

Nothing past that point has a recorded output: the **integration gate** (one full config,
2 epochs, checkpoint-and-reload, two 50-episode shards — the "does resume actually work on
a real config" check), the **ten-configuration training/evaluation matrix**, the local
**results consolidation**, and the **final acceptance-and-teardown check** have not run
yet through this notebook.

**Update, now that Step 9 has closed:** the plain notebook's already-completed run
(Section 1 above) turned out to be recoverable after all — its results made it back into
this repository's `results/` folder and were transcribed into the official results table
and write-up. That resolves the "stranded results" finding this section originally
flagged: the fastest path described below (recover the completed run rather than
re-running the matrix) is the one that actually closed out Step 9. This relay notebook's
own run was not the source of Step 9's numbers and, as far as this update knows, remains
stopped at the data-staging step described above — nothing new is known about its
progress beyond what Section 1 already shows.

## 2. Why we did it

The plain notebook (Sections 1–7 above) runs everything — code, data, training, and the
final results — inside one Kaggle or Colab session. That works, but it has two rough
edges this addendum's notebook exists to remove: first, if that single session
disconnects partway through a long run (and Step 9's own instructions.txt budgeted ~2–2.5
hours for it), you can lose everything since the last manual save; second, even a fully
successful run's results still have to be zipped up and carried back out of that session
by hand (or by a best-effort automated delivery step) before they're usable here — which,
as it turned out, is exactly what happened with Step 9's real run: the results sat on
Kaggle's side until a separate recovery step brought them back. The relay notebook's whole
design is to make both of those problems structurally harder to hit: every unit of remote
work (a data chunk, an epoch, an evaluation shard) is individually hash-checked and
independently resumable, and finished results are written directly into this repo rather
than accumulating somewhere that still needs to be fetched.

## 3. The analogy

Picture the same renovation crew from Section 3 above, still working on the house in the
new neighborhood — but this time, instead of flying the whole crew out to live on-site for
the entire job and mail back a single box of photos and receipts at the very end, the
general contractor stays at the home office and works the job by phone: each call sends
the out-of-town crew exactly one task — "pour this slab," "frame this wall," "inspect this
room" — and the crew reports back and gets checked off before the contractor makes the
next call. If the phone line drops mid-call, only that one task needs to be redialed, not
the whole job restarted from day one. Before trusting the out-of-town crew with the actual
blueprints, the contractor first places one cheap test call — "can you lift this one test
board and hang up?" — and only proceeds once that comes back clean. And rather than
waiting for one big box of paperwork to be mailed back at the end, every receipt is filed
straight into the home office's own cabinet the moment each task is confirmed done — which
is a better system than what actually happened with the other crew's box, which sat at the
depot for a while before someone went and picked it up.

## 4. Mapping analogy to reality

| Analogy element | Real technical thing |
|---|---|
| The home office, holding the paperwork/records/blueprints | This repo's local checkout — source, configs, `results/`, `checkpoints/` |
| The out-of-town crew, hired only for specific tasks | The temporary, checksum-pinned worker running on a borrowed Kaggle Tesla T4 |
| One cheap test call before trusting the crew with real work | The verified relay probe from `docs/gpu-relay-guide.md` (real CUDA matmul, real teardown, must print `VERIFIED`) |
| Each phone call being one specific, checked-off task | One remote job per training epoch; one remote job per 50-episode evaluation shard |
| Redialing just the dropped call, not the whole job | Chunked, SHA-256-checked, resumable source/data staging (demonstrated by two deliberate self-interrupt-and-resume tests) |
| A trial task before trusting the crew with the whole house | The integration gate: one full config, 2 epochs, checkpoint export + reload, two small evaluation shards, before the other 9 configs unlock |
| Filing each receipt straight into the home office cabinet | Results/checkpoints written directly into this repo's `results/`/`checkpoints/` as each piece finishes, instead of zipped up for later delivery |
| The other crew's box of paperwork, which sat at the depot before eventually being picked up | `step9-mini.ipynb`'s already-completed 10-run results — initially packed into a Kaggle-side zip, later recovered into this repo's `results/` folder |

## 5. Benefit / what it improved

There are still no accuracy, calibration, or OOD numbers to report *from this notebook
specifically* — this run hasn't reached the training matrix, and Step 9's real numbers
(Section 5 above) came from the other path. What this notebook did concretely
demonstrate, end to end, before stopping:

- A full local pytest run (213 tests) gates any GPU spend at all — nothing is sent to the
  remote worker unless the whole repo's test suite already passes locally.
- The live relay path genuinely works: real authentication, real node selection, a real
  CUDA workload on a real Tesla T4, and a confirmed teardown, exactly as
  `docs/gpu-relay-guide.md` describes.
- The resumability design isn't just a claim — the notebook proved it against itself twice
  (one source chunk, one data chunk) by deliberately interrupting and then continuing from
  the exact byte offset the worker reported.

## 6. How this fits the overall thesis

This addendum doesn't add a new research finding — it's plumbing, not science. Its
contribution is to Step 9's *reliability*, not its results: this project's own
reproducibility rules (CLAUDE.md: byte-identical reruns, frozen episode seeds, honestly
reported state) are easiest to keep once results land in one predictable local place as
soon as they exist, rather than depending on a separate manual step to retrieve them from
wherever they were computed. As it happened, Step 9's real numbers came from the plain
notebook plus a manual recovery step, not from this relay path — but nothing about how
those numbers are interpreted changes either way: same configs, same frozen test seeds,
same 600 episodes.

## 7. What's next, and why

Step 9 itself is closed, and its numbers came from recovering the plain notebook's
already-completed run rather than from finishing this relay run. This notebook's own
progress (stopped mid-data-staging, Section 1) is not currently blocking anything — Step
10 is a large new grid of runs, not a rerun of Step 9 — so whether to pick this notebook
back up is now a question of whether the relay's resumability is worth relying on for
Step 10's much larger run count, not something Step 9 itself still needs.
