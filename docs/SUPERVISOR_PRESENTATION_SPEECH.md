# B-PEFT — Presentation Speech Script

**Companion to:** `docs/SUPERVISOR_PRESENTATION_SLIDES.md` (one section per slide)
**Target:** ~25 minutes of speaking + questions
**Tone rule:** state the number, then state the caveat, then move on. Do not oversell, do not
apologise. If you do not know something, say "that is not measured — it is on the backlog."

**Three sentences to have memorised, in case you get lost:**
1. "A 6,928-parameter adapter reaches the accuracy of retraining an 11.18-million-parameter network."
2. "The evidential head made OOD detection much better and calibration much worse — those turned out to be separate skills."
3. "All four research questions have numerical answers; two passed, one failed cleanly, one split."

---

## SLIDE 1 — Title *(~30 sec)*

> "Thank you for the time. This is B-PEFT — Bayesian Parameter-Efficient Fine-Tuning for reliable
> few-shot vision on lightweight CNN backbones.
>
> Where we stand today: eleven of thirteen implementation steps are closed, and as of August ninth
> **all four research questions have numerical answers**. The evidence base is 120 completed training
> runs, zero errors, 36.3 GPU-hours.
>
> I want to walk you through the four verdicts honestly — two passed, one failed cleanly, one is
> split — then what is genuinely novel versus what is borrowed, then the limitations I already know
> about, including the one instruction of yours that is not yet done. I would like to end by asking
> you to pick our next move."

---

## SLIDE 2 — The problem *(~1 min 30)*

> "The setting is an edge device that has to learn five new classes from five example images each,
> and cannot retrain a large model because it does not have the memory.
>
> Almost everyone in this space reports one number: accuracy. But on a device with nobody watching,
> three other things matter just as much. When the model says it is ninety percent sure, is it
> actually right ninety percent of the time — that is calibration. Does it notice when it is shown
> something it has never seen — that is out-of-distribution detection. And does it actually fit and
> run on the hardware.
>
> The thesis question is narrow and testable: **if you adapt a frozen lightweight CNN using only a
> few thousand trainable parameters, does the uncertainty machinery still work — or does it quietly
> break?** The honest answer we found is: partly. And which part breaks turned out to be the
> interesting result."

---

## SLIDE 3 — What we built *(~1 min 30)*

> "The pipeline is three pieces: a frozen CNN backbone, a tiny trainable adapter, and a head.
>
> The backbone — ResNet-18 or MobileNetV3-Small, both ImageNet-pretrained — is completely frozen.
> Zero parameters updated. The adapter is the only thing that learns, and depending on the method
> that is between **6,928 and 31,746 parameters** — between 0.06 and 0.28 percent of the backbone.
> The head is parameter-free; it classifies by comparing to prototypes built from the support set.
>
> The Bayesian piece is one design decision. A standard softmax is *mathematically forced* to output
> a confident distribution, even for complete nonsense — it has no way to say 'nothing here looks
> familiar.' The Evidential Dirichlet head can represent exactly that state. That mechanism is what
> we set out to test, and it is what RQ2 and RQ3 are about."

---

## SLIDE 4 — The four research questions *(~1 min 30)*

> "These are the four questions as written in the proposal. Adapter placement. Whether an evidential
> head calibrates better than softmax. Whether a Bayesian prior improves near-OOD detection in low
> data. And the latency-versus-uncertainty Pareto frontier.
>
> Two scope notes I want to make before the results, rather than after.
>
> **First** — RQ2 as written says 'under 500 trainable parameters.' Only one of our cells literally
> meets that: the linear probe, at **two** parameters. Our PEFT adapters sit between roughly seven
> and thirty-two thousand. We do report the sub-500 case, and it behaves the same way, but our
> headline RQ2 answer covers the whole range. That is a widening of the proposal's wording and I do
> not want to gloss over it.
>
> **Second** — RQ1 got answered in two parts. The literal serial-versus-parallel question was settled
> in Step 6. The main grid then took the winner and compared it against LoRA. Both results are on the
> next slide."

---

## SLIDE 5 — The evidence base *(~2 min — take your time here)*

> "This slide is what makes everything after it credible, so I will be specific.
>
> Five axes. Two datasets — CIFAR-FS and MiniImageNet. Two shot regimes — one-shot and five-shot,
> five-way throughout. Two backbones. Four adapters. Two head interpretations. That is **forty unique
> configurations**. Each was trained with **three seeds**, so **120 training runs** in total. Each run
> was evaluated on the same **600 frozen test episodes**.
>
> The run log is 123 'ok' plus one 'skipped-done', **zero errors**. The aggregated results file
> reports 120 of 120 cells present, 40 of 40 groups with all three seeds, and an empty missing-cells
> list. Total wall time 36.3 hours.
>
> On top of that, every run was evaluated against **five OOD pools** — SVHN and Gaussian noise as
> far-OOD, and CIFAR-100-heldout, MiniImageNet-heldout and TinyImageNet as near-OOD.
>
> So when I say something like 'twenty out of twenty' later, that is twenty genuinely matched
> comparisons where only one thing changed, not twenty reruns of the same setup."

---

## SLIDE 6 — Why the numbers can be trusted *(~1 min 30)*

> "Four guarantees, quickly.
>
> The 600 test episodes are **frozen** — the seeds are fixed in a config file that is marked
> do-not-regenerate, so every run in the thesis is comparable to every other run. All hyperparameter
> search ran on a **separate validation seed range**; we never tuned on the test seeds. Reruns are
> **byte-identical** — and we verified that non-trivially: the grid's seed-42 cells reproduced Step
> 6's earlier committed metrics exactly, twenty-nine of twenty-nine and fifty-five of fifty-five
> numeric keys. And for RQ4 the cost and quality axes were **pre-registered** — written down before
> any latency number existed, so they could not be tuned after the fact.
>
> On noise: seed standard deviation exceeds one percentage point in only **two of forty**
> configurations. Everywhere else it is under 0.74. So differences above two points are real.
>
> One exception I will flag now because it affects a later claim: our two baselines — full
> fine-tuning and linear probe — have **zero** seed variance, by construction. Full fine-tuning starts
> from fixed pretrained weights, and the linear probe has no trainable tensor at all, so there is
> nothing for the seed to perturb. For those two rows, n equals three is effectively n equals one."

---

## SLIDE 7 — RQ1 result *(~2 min)*

> "RQ1 passed, and it is the cleanest result in the thesis.
>
> **Part A, the literal proposal question.** Serial versus parallel placement is an accuracy **tie** —
> 0.9145 against 0.9146. The tiebreak came from OOD detection, where parallel is equal or better on
> every pool, with the gap concentrated in far-OOD. And both in-block placements beat the post-pool
> placement by three to four points.
>
> I want to flag something here rather than let it be found. When I checked this more carefully, the
> closest prior work turned out to be closer than I first thought. There's a CVPR 2022 paper called
> Task-Specific Adapters — TSA — that runs this exact comparison: frozen ResNet-18, serial versus
> parallel adapter connection, six hundred sampled episodic tasks on held-out domains, a
> parameter-free nearest-centroid head. Same backbone, same episode count, same question, and they
> find the same answer — parallel wins in almost all cases. That's a closer match than Conv-Adapter,
> which I'd cited before. I'm not going to present this as a discovery; it's a confirmation, and I'd
> rather say so myself than have it pointed out.
>
> **Part B, the winner against LoRA across the full grid: sixteen out of sixteen matched comparisons,
> between two-point-one and eight-point-three points better. There is not one configuration where
> LoRA wins.** This specific comparison — LoRA against bottleneck adapters — is not something TSA or
> Conv-Adapter runs, so this part is genuinely ours.
>
> And on MobileNetV3-Small it is a **strict Pareto win** — the parallel bottleneck is both cheaper,
> 6,930 versus 10,754 parameters, *and* more accurate. There is no trade-off to negotiate there;
> LoRA is simply dominated.
>
> The interpretation I would put in the thesis: LoRA is a **transformer-native reparameterisation**.
> This is direct evidence it should not be the default choice for convolutional backbones."

---

## SLIDE 8 — RQ1 headline *(~2 min — this is your strongest slide, but land the caveats)*

> "Here is the number I would put on the front page.
>
> On CIFAR-FS five-shot, a **31,744-parameter adapter beats full fine-tuning of an 11,176,512-parameter
> network** — 91.44 against 90.47 — while training 0.28 percent of the parameters. And a
> **6,928-parameter** adapter on MobileNetV3-Small reaches 90.74, which is the same accuracy as
> retraining that eleven-million-parameter network, at 0.06 percent of the budget.
>
> Three caveats, and I want them on this slide rather than a backup slide.
>
> **One** — the MobileNet row **matches** full fine-tuning; it does not beat it. The margin is 0.27
> points, which is smaller than that cell's own seed spread. We phrase it as 'matches' deliberately,
> everywhere in the write-up.
>
> **Two** — and this is the one I most want your input on — that is a **cross-backbone** comparison.
> Full fine-tuning has never been run on MobileNetV3-Small, so the only full-fine-tuning number that
> exists is ResNet-18's. That is the coverage gap on slide twenty.
>
> **Three** — at one-shot the ordering **flips**. Full fine-tuning wins by 2.57 points. With a single
> image per class, the extra capacity buys something a tiny adapter genuinely cannot recover. That is
> the honest limit of the claim."

---

## SLIDE 9 — RQ2 result *(~2 min — deliver this without flinching)*

> "RQ2 was the central hypothesis of this thesis, and **it did not pass**.
>
> The evidential head is worse calibrated than plain softmax in **twenty out of twenty** matched
> pairs — by a factor of 1.4 to 9.1. Against temperature-scaled softmax, by a factor of **5.3 to
> 51.2**. And accuracy does not compensate: evidential is 0.61 points worse on average, winning only
> seven of twenty.
>
> Two worked examples. CIFAR-FS five-shot: evidential ECE 0.3010 against temperature-scaled softmax
> at 0.0152. MiniImageNet five-shot: 0.2938 against **0.0057** — a fifty-one-fold gap.
>
> The obvious objection is 'you did not tune it.' We checked that. A validation-only sweep in Step
> 4.5 found the ECE surface **flat**, around 0.285 to 0.296, across the range tested. The gap does
> not close by tuning.
>
> And one fairness caveat I will volunteer: temperature scaling is a cheap post-hoc fix available to
> softmax, and we have **not** implemented an equivalent for evidential. So that column is
> structurally favourable to softmax. It is still the right comparison — temperature scaling is
> standard and free — but the asymmetry is real, and it is stated in the write-up."

---

## SLIDE 10 — Why RQ2 failed and why it is still worth having *(~2 min 30 — the most important slide in the deck)*

> "So why did it fail, and why am I presenting it as a result rather than an embarrassment?
>
> **The mechanism.** The evidential head has to *learn* a mapping from 'how much evidence do I have'
> to 'how confident should I sound.' We froze the backbone and gave it — in one cell — literally
> **two** trainable parameters, and five images per class. There is not enough capacity or signal to
> fit that mapping. So its stated confidence stops tracking its actual hit rate. It still ranks
> classes correctly, which is why accuracy is fine, but the *number* attached to that ranking is
> meaningless.
>
> **Three reasons this is worth having.**
>
> First, it is **well-powered**. Two datasets, two shot regimes, two backbones, four adapters —
> twenty matched pairs, zero exceptions. That is not a single-configuration fluke that someone can
> wave away.
>
> Second, **independent theory agrees**. A 2026 analysis shows that standard reverse-KL evidential
> objectives retain non-vanishing epistemic uncertainty *even in the infinite-data limit*. We did not
> know that paper when we ran the grid. An experiment and a proof arriving at the same conclusion
> from different directions is much stronger than either alone — and it rules out 'your code has a
> bug' as the explanation.
>
> Third — and this is the part I would defend hardest — **it locates a boundary, and now from two
> independent directions, not one**. The nearest prior work, BEL, found evidential calibration
> *improving*: 3.59 percent against a 14.69 baseline. That is the opposite of our result. But BEL
> **meta-trains the entire backbone**, and we **freeze** it. When I went and checked further, I found
> a second, independent paper — BayesAdapter, published in IJCV in 2025-26 — using a completely
> different Bayesian mechanism, variational Bayes over the weights of a linear adapter, not an
> evidential Dirichlet head at all. On a frozen CLIP backbone with up to thirty-two shots, it also
> finds calibration *improving*, by about two and a half percent ECE. So that is two independent
> papers, two different Bayesian mechanisms, both improving when there's more capacity or data to
> work with than our grid provides. Both results are true; they are different regimes. So we did not
> fail to reproduce BEL, or BayesAdapter — we established where the boundary sits, and it's now
> triangulated from two directions instead of one. Bayesian calibration needs capacity or data to
> work with, and it breaks once you freeze the backbone and cap the budget as low as two parameters.
>
> And that is precisely the edge-deployment side of the boundary — because meta-training a backbone
> is exactly what a 256-kilobyte device cannot do."

---

## SLIDE 11 — RQ3 result *(~2 min)*

> "RQ3 is split, and I will give you the win first and the loss immediately after.
>
> Against every softmax-probability-based score, the evidential uncertainty is a decisive win.
> Against max-softmax-probability: thirty-eight of forty on far-OOD, thirty-seven of forty on
> near-OOD — **93.8 percent overall**. Against temperature-scaled MSP, **95.0 percent**.
>
> And the specific RQ3 prediction — that the Bayesian prior helps *more* when there is *less* data —
> **held**. The near-OOD advantage over MSP is **plus 0.064 at one-shot against plus 0.043 at
> five-shot**. Over temperature-scaled MSP, 0.073 against 0.061. That is the low-data trend the
> hypothesis predicted, and it survived contact with the full grid rather than being a one-off."

---

## SLIDE 12 — RQ3's counterpoint *(~2 min — volunteer this energetically; do not let it be extracted)*

> "Now the part I want to raise myself.
>
> There is a training-free score called the **energy score** that beats our Bayesian one in about
> **seventy percent** of comparisons. It needs no evidential training, no Bayesian prior, no extra
> parameters — it is computed from the same raw logits a plain softmax model already produces. Across
> the grid, our vacuity score wins only ten of forty on far-OOD and fourteen of forty on near-OOD
> against it.
>
> And this **corrects our own earlier finding.** Step 4.5, on a single configuration, concluded
> evidential was roughly on par with energy. When we scaled from one configuration to forty, that did
> not generalise. The correction is written into the repository's decisions log and into the project
> instructions file, dated August sixth, specifically so nobody — including us — cites the old claim
> by accident.
>
> So the defensible claim is narrower than 'the Bayesian prior improves OOD detection.' It is:
> *among scores derived from the model's own predictive distribution*, Dirichlet vacuity is a
> substantially better ranker than max-softmax-probability with or without temperature scaling, and
> its advantage grows in the lowest-data regime — **but it does not beat a well-chosen logit-space
> score.**
>
> If you asked me what to tell a practitioner today: for OOD detection alone, use softmax plus the
> energy score. Evidential is the better *native probabilistic* option, not the best option overall."

---

## SLIDE 13 — Calibration and OOD are decoupled *(~1 min 30)*

> "This is the cross-cutting finding, and it is the one I find most interesting.
>
> The **same head** that is five to fifty-one times worse calibrated is **simultaneously** the better
> OOD detector ninety-four percent of the time. That looks contradictory until you notice the two
> metrics are asking different questions.
>
> Calibration demands that the confidence **number** be honest — that is a magnitude requirement, and
> it needs a correctly-fitted evidence-to-confidence mapping, which our parameter budget cannot
> afford. OOD detection only demands that unfamiliar inputs score **lower** than familiar ones — that
> is a ranking requirement, and it survives a badly-scaled mapping just fine.
>
> Same broken mapping, two completely different consequences. These properties are routinely treated
> as one thing — 'uncertainty quality' — and our data says they are not. Reporting ECE alone would
> have hidden this. So would reporting AUROC alone. We only see it because both were measured on the
> same 120 runs."

---

## SLIDE 14 — RQ4 result *(~1 min 30)*

> "RQ4 closed on August ninth, on real hardware — Kaggle T4 GPU plus Kaggle CPU — with the axes
> pre-registered before any number existed.
>
> The recommended operating point is **MobileNetV3-Small plus parallel bottleneck plus the evidential
> head**: 11.86 milliseconds per image on a single CPU thread, 6,930 trainable parameters, near-OOD
> AUROC of 0.870 at one-shot and 0.919 at five-shot.
>
> On MiniImageNet it shifts to ResNet-18, at 62.38 milliseconds, because MobileNetV3-Small falls
> outside the pre-registered accuracy tolerance on that dataset — which is consistent with the
> backbone gap we already reported widening specifically on MiniImageNet.
>
> One small honesty note on this table: our measured deployed parameter count for MobileNetV3-Small
> is 933,938, not the 2.5 million you see in the literature. The difference is the ImageNet classifier
> head, which our stack does not deploy. We report our own measurement."

---

## SLIDE 15 — RQ4's novel finding *(~2 min)*

> "The most novel result from Step 11 is this: **backbone choice drives latency, and adapter choice
> essentially does not.**
>
> ResNet-18 against MobileNetV3-Small at a matched adapter is a **5.12-times** latency difference —
> 62 milliseconds against 12. But the parallel bottleneck against LoRA on ResNet-18 — which is a
> **2.58-times difference in trainable parameters** — is only a **3.9 percent** latency difference.
> And on MobileNetV3-Small, the adapter with **more** parameters is actually **5.8 percent faster**.
> Both adapters' parameter deltas are completely swamped by the frozen trunk's forward-pass cost.
>
> That gives a deployer a clean rule: **the adapter decision is an accuracy decision — up to 8.3
> points. The backbone decision is the latency decision — 5.12 times.**
>
> And the claim everyone makes but nobody measures — that evidential uncertainty is free at inference
> — we **measured** it. Mean latency delta between evidential and softmax heads at matched backbone
> and adapter: **1.29 percent**, which is *below* this session's own measurement-noise floor of 5.91
> percent. The whole uncertainty-scoring stage for a 75-query episode costs one percent of a single
> image's backbone forward pass.
>
> One conditional, though. Under our primary reading — each head scored with its native score —
> evidential heads anchor every strict frontier. But under a 'softmax gets its best score' reading,
> where softmax is allowed the energy score, evidential's frontier presence on CIFAR-FS five-shot
> **collapses to zero**. So the Pareto recommendation is native-score-conditional, not universal. That
> is the same energy correction from RQ3 showing up again in a different table."

---

## SLIDE 16 — Scorecard *(~1 min)*

> "So, the scorecard. RQ1 passed — sixteen out of sixteen, zero exceptions. RQ2 failed — zero out of
> twenty. RQ3 is split — ninety-four percent against softmax scores, thirty percent against energy.
> RQ4 passed, measured on real hardware with pre-registered axes.
>
> Two passes, one clean negative, one split.
>
> I would rather present this than four confirmations. A thesis where every hypothesis came back
> 'yes' would suggest the questions were chosen safely rather than tested honestly. And the negative
> on RQ2 is the result that most changes what the next researcher should do — it stops the next group
> spending a year rediscovering it."

---

## SLIDE 17 — Novelty *(~2 min 30 — expect the "hasn't this been done?" challenge here)*

> "Let me be direct about novelty, because I expect the question.
>
> **We invented none of the individual pieces, and I will say so first.** Evidential Dirichlet heads
> are Sensoy et al., 2018. Bottleneck adapters and LoRA are existing PEFT methods we borrowed.
> Episodic few-shot training and prototype networks are standard since 2017. Freezing a pretrained
> backbone is ordinary transfer learning. The energy score, temperature scaling and ECE are all
> established. If someone says 'I have seen evidential learning before,' they are right, and agreeing
> costs me nothing.
>
> **Four things are novel, and I will give them to you ranked by strength, not in RQ order — lead
> with the first one if you ask me what's novel, not the weaker claim at the end.**
>
> First, **a well-powered negative that contradicts two independent prior papers, and explains the
> contradiction.** BEL and a second paper called BayesAdapter — different Bayesian mechanisms,
> evidential Dirichlet versus variational Bayes over linear weights — both say calibration improves;
> we say it degrades twenty out of twenty; the difference is capacity and data versus our frozen,
> extreme-low-budget regime. We located a boundary from two directions, not one. This is the single
> strongest claim in the thesis.
>
> Second, **calibration and OOD-ranking quality are empirically decoupled** — shown on the same 120
> runs. Measuring only one metric would have hidden this entirely.
>
> Third, **we overturned our own earlier claim** when we scaled from one configuration to forty —
> evidential does not hold up against the energy score at scale. We wrote the correction into the
> repository rather than quietly leaving the old number in the documents.
>
> Fourth, **backbone drives latency and adapter does not**, and 'evidential is free at inference' is
> now a measured 1.29 percent rather than an assumption.
>
> There is a fifth claim, and I want to be precise about how much weight it can carry, because I
> checked it and it is weaker than I first thought — and the closest match turned out to be closer
> than I first found, too. **The 'unstudied regime' claim** — frozen lightweight CNN, parameter-free
> prototype head, trainable budget down to two, disjoint-class episodic testing, calibration and OOD
> and parameters and latency together. The primary work by *idea* is a CVPR 2022 paper called
> Task-Specific Adapters, not Conv-Adapter — it runs the serial-versus-parallel comparison on a
> frozen ResNet-18 with a parameter-free head over six hundred sampled episodic tasks, the same
> backbone, protocol, and head design as ours, already covered on slide seven. That's a confirmation,
> not a discovery. FiT is a second, close in parameter scale — down to about eleven and a half
> thousand parameters — but with a different adapter design and no placement comparison. And
> Tip-Adapter and CLIP-Adapter cover the general 'freeze a backbone, add a small adapter, do
> few-shot' recipe. I read all of these in full before making this claim, specifically because the
> recipe is close enough that assuming would have been sloppy. What I found for Tip-Adapter and
> CLIP-Adapter specifically: both train and test on the *same* classes — Tip-Adapter's own paper
> contrasts itself directly with meta-learning protocols that split into disjoint category subsets —
> and both report one fixed-test-set accuracy rather than an average over sampled episodes, report no
> calibration or OOD metric, and use adapters seventeen times to orders of magnitude larger than our
> range. So across all of them, the recipe is shared. A budget down to two parameters, a second
> backbone, the LoRA-specific comparison, and the calibration and OOD measurement are not. Say
> 'checked, not discovered' if this comes up — not 'nobody has done this,' because the honest version
> is narrower than that."

---

## SLIDE 18 — The literature gap *(~2 min)*

> "This table is why that intersection was empty. Eight literatures, each covering part of the
> problem, none covering the middle.
>
> Classical few-shot reports accuracy only. Foundation-model few-shot the same, on backbones that do
> not fit an edge device. PEFT for vision transformers reports accuracy and parameter budget, and is
> not few-shot at all. PEFT for CNNs on edge reports accuracy and parameters, no uncertainty. There's a
> row I added after checking further — frozen-CNN episodic few-shot adapters, TSA and FiT — accuracy
> and parameters, no uncertainty either. Bayesian PEFT reports calibration — all but one of those
> papers is on **language models**; the exception, BayesAdapter, is vision but not edge. Evidential
> few-shot — that is BEL — reports calibration but no OOD and no parameter accounting, and meta-trains
> the backbone. TinyML reports efficiency and nothing about uncertainty.
>
> Two of those crosses are **verified absences**, not our inference. The survey covering the entire
> ViT-PEFT field never discusses calibration, uncertainty quantification, or OOD detection — anywhere.
> The 2025 TinyML-to-TinyDL survey covers quantisation extensively and never mentions uncertainty. Two
> independent communities, the same blind spot.
>
> One caveat on my own table: the LoRA-C/LoRA-Edge/CoLoRA part of the PEFT-for-edge row was checked
> against abstracts and summaries, not by exhaustively reading every table in those papers. That is
> flagged in the results document as needing verification before it goes into the written thesis. I
> do not want to claim more rigour there than I have.
>
> Three cells in this table I want to call out specifically, because I checked them properly rather
> than assuming from category, and the check moved the answer — one of them moved it more than I
> expected. TSA, Conv-Adapter, and Tip-Adapter and CLIP-Adapter are all marked 'partial,' not a clean
> checkmark, on few-shot-episodic. TSA is actually genuinely episodic — six hundred sampled tasks on
> held-out domains — and it's marked partial because of what it doesn't report, not its protocol; it's
> the row I'd point to first if asked for the closest prior work on placement. Conv-Adapter does
> low-shot fine-tuning with a trainable head, not disjoint-class meta-testing. Tip-Adapter and
> CLIP-Adapter train and test on the same classes — Tip-Adapter's own paper says so directly,
> contrasting itself with meta-learning splits — and report one fixed accuracy number, not an
> episodic average. I read all of these papers in full before writing this table, because the general
> recipe they use is close enough to ours that I did not want to hand-wave it.
>
> And on whether the question is dated: there are at least six Bayesian-PEFT papers in the last twelve
> months, plus a dedicated **benchmark** paper in 2026. Fields do not build benchmarks for dead
> questions. Almost all of them are on language models — one, BayesAdapter, is vision, but on CLIP, not
> an edge-deployable backbone. None is few-shot vision on an edge backbone."

---

## SLIDE 19 — Positioning and the cost of efficiency *(~2 min 30)*

> "This is the comparison that matters, and it is fair — P>M>F runs the same five-way few-shot
> episodic protocol on the same two datasets. The difference is that they meta-train and fine-tune
> the whole backbone; we freeze it and train an adapter.
>
> Three conclusions, and they are separate — I do not want to blur them into one.
>
> **One: at five-shot on CIFAR-FS, the parameter saving is nearly free.** Our 31,744-parameter adapter
> is **1.06 points** behind a fully meta-trained DINO ViT-S while training **662 times fewer**
> parameters, and 0.76 behind ViT-B at 2,703 times fewer. That is the strongest trade in the thesis.
>
> **Two: backbone-family-matched, we win outright.** Against DINO-ProtoNet on ResNet-50 we are
> **plus 3.56 points** on MiniImageNet five-shot and **plus 5.83** at one-shot, at 788 times fewer
> trainable parameters. So the gap that remains against the ViTs is a **ViT gap, not a
> parameter-efficiency gap**.
>
> **Three, and I want to volunteer this: the trade is genuinely bad at one-shot on MiniImageNet with
> the small backbone — minus 18.2 points.** The 6,928-parameter cell is competitive on CIFAR-FS and it
> is **not** competitive on MiniImageNet. I would not present that cell as universally good.
>
> And one caveat that applies to **every** accuracy number I have shown: our backbones are
> ImageNet-pretrained, and MiniImageNet's classes **are** ImageNet classes — so every 'novel' test
> class was seen during pretraining. Our MiniImageNet numbers are not few-shot results in the sense
> the benchmark was designed to measure. This is not a discovery of ours; the P>M>F authors say the
> same about their own supervised-ImageNet row, and put it in supplemental material for exactly this
> reason. The fix is a from-scratch control run, and it is on the next slide."

---

## SLIDE 20 — Limitations *(~2 min 30 — slow down, this is where you earn credibility)*

> "Eight limitations, ordered by how badly a reviewer would want them fixed. I would rather raise
> these than have you find them.
>
> **Number one, and I want to address this directly: your instruction from August sixth — that every
> combination should be tested — is not yet done.** Full fine-tuning and linear probe have real
> three-seed coverage on **ResNet-18 and CIFAR-FS only**. Full fine-tuning has never run on
> MiniImageNet on any backbone. Neither baseline has ever run on MobileNetV3-Small, on either dataset
> — I confirmed that by direct search; no config file for it exists anywhere in the repository. That
> is why the MobileNet-matches-full-fine-tuning claim is a cross-backbone comparison. It is scoped as
> Step 12.F: twenty-four configs, seventy-two runs, about eighteen to nineteen GPU-hours. We
> deliberately deferred it until Step 11 closed, because Step 11 answered the one research question
> with no answer at all, while 12.F fills a coverage gap in a story that was already complete. That
> condition is now met, so it is ready to run.
>
> **Two, the ImageNet-pretraining confound** I just described — the fix is a from-scratch control run,
> and it is the single highest-value scientific addition.
>
> **Three, no transformer arm in our own grid.** We argue from published ViT numbers, not from a ViT
> we trained. Step 11 gives us a real architecture-only measurement for DeiT-Tiny and ViT-B, but not a
> trained accuracy number.
>
> **Four, three seeds — and effectively one for the two baselines**, as I mentioned. Any claim resting
> on a margin under one point is thin, and we say so on every affected claim.
>
> **Five**, one frozen recipe across all 120 cells. That was deliberate — it makes the grid a
> controlled comparison rather than forty separately-tuned numbers — but it means we answer 'how do
> these axes compare under one recipe,' not 'what is the best achievable number per cell.'
>
> **Six**, evidential has no post-hoc calibration fix in our codebase, so the temperature-scaling
> comparison is structurally favourable to softmax.
>
> **Seven**, there is an inherited open issue: a TinyImageNet class-exclusion check was traced by code
> inspection but never confirmed by a logged exclusion count from a real run. Every TinyImageNet
> near-OOD number carries that caveat.
>
> **Eight**, the proposal's scope was trimmed. ConvNeXt-Nano, CUB-200, ISIC, CIFAR-10-C, and BitFit in
> the grid are all deferred to Step 12. That was a depth-over-breadth decision — we prioritised getting
> all four research questions a numerical answer. They are listed explicitly in the backlog, not
> dropped silently."

---

## SLIDE 21 — The two bugs *(~1 min 30 — deliver this confidently; it builds trust)*

> "One more thing I want to report rather than have discovered. Two real bugs were found during Step
> 11's closeout.
>
> The first crashed loudly in an optional feature and discarded a correct measurement before it could
> save. Fixed at the root.
>
> The second is the one worth your attention, because it was **silent**. Three independent
> latency-selection functions, in three different scripts, all picked our own noisy dev-laptop latency
> instead of the canonical Kaggle CPU number — by a dictionary-insertion-order accident. **No crash. No
> failing test.** Errors up to forty-seven percent on individual cells. It was caught only by a manual
> cross-check during closing verification.
>
> Impact: every panel's **recommended** Pareto point was unaffected. **Frontier membership** was
> affected — with corrected data, both MobileNetV3-Small LoRA variants are genuinely on the CIFAR-FS
> frontier, where previously one appeared dominated. Everything downstream — Table 8, the frontier
> JSON, all six Pareto figures — was regenerated from corrected code.
>
> And the open action item: **that bug still has no regression test.** It is the top item on our
> backlog, because the bug class that produces no error at all is exactly what a test suite exists to
> catch."

---

## SLIDE 22 — Close and ask *(~1 min 30)*

> "So — eleven of thirteen steps closed, all four research questions answered, 120 runs with zero
> errors and fully reproducible results.
>
> What I would like from this meeting is a decision on what comes next. There are five candidates.
>
> Writing the regression test for that silent bug — cheap, and it protects every number we produce
> from here. Running 12.F, your instruction, which would remove the cross-backbone caveat from our
> headline RQ1 claim — about eighteen GPU-hours. The from-scratch control run, which would remove the
> pretraining objection entirely and is the highest scientific value. A trained ViT-Tiny arm, which
> would convert our 'CNNs are right for edge' argument from a literature citation into our own
> measurement. Or starting Phase 6, the thesis writing.
>
> **Our recommendation, for you to challenge:** the regression test first because it is cheap and
> protects everything, then 12.F because it is your instruction and it hardens the headline claim —
> and start the writing **in parallel**, because no research question is blocked on any of this. The
> from-scratch control and the ViT arm are the highest-value additions if the timeline allows, in that
> order.
>
> I am happy to go deeper on any of the four results, or on the limitations."

---

## Q&A PREP — likely questions and short answers

| Question | Answer |
|---|---|
| **"Hasn't all this been done before?"** | "Every ingredient, yes — and I say that on slide seventeen. The strongest thing that's new isn't the combination of ingredients, it's that we resolved a real contradiction, and now from two directions: BEL and BayesAdapter — two different Bayesian mechanisms — both say calibration improves, we found it degrades, and we located exactly why — capacity and data versus our frozen, extreme-low-budget regime. That plus the calibration/OOD decoupling are the two claims I'd defend hardest. The 'nobody's tested this regime' framing is real but narrower than it sounds — I checked TSA, FiT, Tip-Adapter and CLIP-Adapter directly rather than assuming, and TSA in particular is closer than I first found: same backbone, same episodic protocol, same placement question, same answer. What's not shared with any of them is our parameter budget, the LoRA-specific comparison, and the calibration/OOD measurement." |
| **"Isn't 'serial vs. parallel adapter placement' just Task-Specific Adapters (TSA)?"** | "Yes, and it's a closer match than the Conv-Adapter citation I had before — I read TSA in full to check. Frozen ResNet-18, six hundred sampled episodic tasks on held-out domains, a parameter-free nearest-centroid head, and it finds parallel wins — same backbone, protocol, question, and answer as our Step 6. What's not in TSA: our parameter budget goes down to two, versus their 175K to 1.22 million; they don't test a second backbone; and they never compare LoRA specifically or report calibration or OOD." |
| **"Isn't 'freeze a backbone, add a small adapter, do few-shot' just Tip-Adapter/CLIP-Adapter?"** | "The general idea, yes — I read both in full to check, not assumed. Three real differences: they train and test on the same classes, not disjoint ones — Tip-Adapter's own paper says so; they report one fixed-test-set number, not an episodic average over sampled tasks; and neither reports ECE or OOD-AUROC, and their adapters are seventeen times to orders of magnitude larger than ours. Shared idea, different protocol and scale." |
| **"Isn't a CNN backbone outdated in 2026?"** | "Transformers are the strongest backbones at scale — I don't dispute that. But an 86-million-parameter ViT does not fit in 1 MB of flash under any quantisation. A 2025 survey measures memory-optimised transformer attention at ~180 ms on an STM32F746 versus 8–12 ms for CNN inference. And an October 2025 study finds CNNs match ViTs specifically in low-data regimes — few-shot *is* the low-data regime. Full answer is in `docs/DEFENCE_BRIEF.md`." |
| **"Is a negative result publishable?"** | "At this coverage, yes — twenty matched comparisons, zero exceptions, two datasets, two shot regimes, two backbones, four adapters, plus independent 2026 theory pointing the same way. It's the result that stops the next group repeating the experiment." |
| **"Why does a free energy score beat your Bayesian one?"** | "Because energy reads the raw logit strength directly, while vacuity reads a *learned* transformation of it — and our parameter budget can't fit that transformation well. Same root cause as the RQ2 calibration failure." |
| **"Why didn't you run CUB-200 / ISIC / ConvNeXt?"** | "Time budget, and it was a deliberate depth-over-breadth choice. The 120-run core grid at 36.3 hours was prioritised so every RQ would get a numerical answer. They are explicit Step-12 backlog items, in a documented drop order." |
| **"Are three seeds enough?"** | "For differences above two points, yes — seed std exceeds one point in only two of forty cells. For sub-one-point margins, no, and we flag every claim that rests on one. And for the two baselines the seed axis is inert by construction, so those are single measurements." |
| **"Did you tune on the test set?"** | "No. All hyperparameter search ran on validation seeds 10000–10099. The 600 test seeds are frozen in a do-not-regenerate config, and reruns are byte-identical — verified against Step 6's committed metrics, 29/29 and 55/55 numeric keys." |
| **"What's the one thing you'd fix with more time?"** | "The from-scratch control run — it removes the ImageNet-pretraining confound, which is the objection that touches every accuracy number in the thesis." |
| **"Why did you defer my 'test every combination' instruction?"** | "Because Step 11 answered RQ4, which had no answer at all, while 12.F fills a coverage gap in an RQ1–RQ3 story that was already complete. The decision and the reasoning are dated in the log. Step 11 is now closed, so 12.F is unblocked and ready — it's about eighteen GPU-hours." |

---

## DELIVERY NOTES

- **Do not rush slides 9, 10 and 12** — the two failures and the correction. Those are where your
  credibility is won or lost. Slow down, state the number, state the caveat, move on.
- **Slide 8 is your strongest** — but land all three caveats. A claim with its limits attached is
  much harder to attack than a bare claim.
- **When you don't know:** "That is not measured — it is on the backlog" is a complete answer. Do not
  improvise a number.
- **If challenged on the energy result:** agree immediately. "Yes — for OOD detection alone, softmax
  plus energy is the better default. That's in the write-up, and it corrects our own earlier claim."
- **Have `docs/RESULTS_MASTER.md` open on your laptop** — it has all 120 runs in full tables if
  anyone wants to see a specific cell.
