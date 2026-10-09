# Neural Signal ML: what each method does, and what worked

Two decoding problems, each with a classic baseline and a neural network, evaluated on data the model never saw.

1. **Motor imagery from EEG** (BCI Competition IV 2a). Nine people imagined moving their left hand, right hand, feet or tongue, 288 times per session, on two different days. The task is to decode which of the four they imagined from 22 scalp electrodes.
2. **Hand velocity from motor-cortex spikes** (Neural Latents Benchmark MC_Maze Medium). A monkey made 250 reaches around a maze while 152 neurons in motor and premotor cortex were recorded. The task is to decode 2D hand velocity from the spikes.

All numbers come from `results/Results.md`, which is generated from the saved runs.

---

## 1. Motor imagery EEG

### The signal

Imagining a hand movement makes the 8–13 Hz **mu rhythm** over the *opposite* motor cortex shrink. This is event-related desynchronisation (ERD), and it's the signal every decoder here is really reading:

![ERD spectra](figures/eeg-erd-spectra.png)

For subject 3, mu power over C3 (left motor cortex) is lower when imagining the *right* hand, and over C4 (right motor cortex) it's lower for the *left* hand.

### Cleaning

![Preprocessing](figures/eeg-preprocessing.png)

- **Eye artefacts.** Blinks and eye movements leak into frontal electrodes. Each 2a session starts with runs where the subject deliberately moves their eyes. We fit a least-squares map from the 3 EOG channels to each EEG channel on those runs, then subtract the predicted eye contribution everywhere (top panel: grey raw vs blue cleaned at Fz).
- **Band-pass.** A 4th-order Butterworth filter: 8–30 Hz (mu + beta) for CSP, and a wider 4–40 Hz for EEGNet so it can learn which frequencies matter. The filter is *causal*, as an online BCI must be.
- **Epoching.** We take 0.5–2.5 s after the cue (shaded), skipping the first 0.5 s where the visual response to the cue dominates.
- **Bad trials.** Trials with expert artefact marks or a peak-to-peak above 100 µV are dropped from *training only*. Test trials are never dropped, because dropping hard test trials would flatter the result.

### Method 1: CSP + LDA

**Common Spatial Patterns** learns weightings of the 22 electrodes (spatial filters) whose output variance differs as much as possible between one class and the rest. Mathematically, these are the extreme eigenvectors of the generalised eigenproblem C_class w = λ (C_class + C_rest) w. The log-variance of each filtered signal is a band-power feature. **Linear Discriminant Analysis** (with shrinkage, since 24 features from about 270 trials is a lot) then separates the four classes.

The *patterns* (not the filters) show where on the scalp each class's power change sits:

![CSP patterns](figures/csp-patterns.png)

Left- and right-hand patterns sit over the motor strip, roughly opposite each other. Feet and tongue are less localised, which matches the confusion matrix below, where they're the classes most often mixed up.

### Method 2: EEGNet

EEGNet is a small CNN (about 3k parameters) whose first two layers mirror CSP:
- a temporal convolution that learns band-pass filters
- a depthwise convolution across all electrodes that learns spatial filters for each band

These are followed by pooling (a power envelope), a separable convolution and a linear classifier. It's trained with Adam for a fixed 300 epochs, keeping the weights with the lowest loss on a 20% validation split of the *training* session.

### Evaluation: held-out session and held-out subject

- **Held-out session:** train on a person's day 1 and test on their day 2. EEG drifts between days (electrode placement, impedance, fatigue), so this is harder than cross-validation within one day, which many papers report.
- **Held-out subject:** train on the other 8 people and test on someone the model has never seen. This is the "no calibration" scenario.

Chance is 25%. With 288 test trials, a random classifier exceeds 29.5% only 5% of the time, so that is the real bar.

![EEG accuracy](figures/eeg-accuracy-by-subject.png)

| | Held-out session | Held-out subject |
|---|---|---|
| CSP + LDA | **63.9% ± 17.6** (κ 0.52) | **44.3% ± 14.2** (κ 0.26) |
| EEGNet | **62.3% ± 15.9** (κ 0.50) | **43.2% ± 14.7** (κ 0.24) |

There is no significant difference between EEGNet and CSP + LDA. EEGNet beats the baseline on 4 of 9 subjects in each protocol, and the mean differences (−1.6 points held-out session, −1.1 points held-out subject) are well inside the noise (Wilcoxon signed-rank p = 0.65 and 0.50). The per-subject numbers move together (correlation 0.85 held-out session, 0.96 held-out subject): whoever is easy for one method is mostly easy for the other, though subjects 5 and 7 swing by 14–19 points between methods.

![Confusion matrices](figures/eeg-confusion.png)

**What we learned**
- **Subjects matter more than methods.** Subjects 1, 3, 7, 8 and 9 are decoded at 70–80% by either method. Subjects 2, 5 and 6 sit near 30–45%, a well-known pattern: some people don't produce a clear ERD (so-called "BCI inefficiency").
- **Cross-subject transfer is hard.** Dropping the day-1 calibration data and using other people costs CSP + LDA about 20 points. Spatial patterns differ between heads.
- **A CNN doesn't beat a good spatial filter here.** With 288 trials per person, CSP + LDA already captures the main signal (power changes in the mu and beta bands over motor cortex). EEGNet's first layers are built to learn the same kind of spatial and frequency filters, and with this little data it has nothing extra to exploit, so it ties rather than wins. The neural network pays off in the spike data below, where there are many more channels and a time series to model.
- **A training-loop lesson.** The first EEGNet run used early stopping with patience 60. On subject 2, validation loss sat at ln 4 ≈ 1.386 (chance) for the first ~60 epochs, so training stopped and kept near-random weights, giving 24% test accuracy. Switching to the EEGNet paper's recipe (a fixed budget with a best-validation checkpoint) fixed it. The validation curve alone was enough to diagnose this.

---

## 2. Hand velocity from motor-cortex spikes

### Preprocessing

- **Binning.** Spikes are counted in 50 ms bins.
- **Neural lag.** Each velocity sample is predicted from spikes 100 ms *earlier*, because motor cortex leads movement.
- **Segments.** For every trial we cut a segment from 750 ms before movement onset to 450 ms after. The first 500 ms warm up the decoder's state, and R² is scored on −250 to +450 ms, the NLB window.

Both decoders get identical inputs.

### Method 1: Kalman filter

The Kalman filter models two linear-Gaussian relationships, both fitted by least squares:

- **The hand's dynamics:** x_t = A x_{t−1} + noise
- **How neurons encode the hand:** z_t = H x_t + noise

Here x is the hand's position, velocity and acceleration, and z is the vector of spike counts. Decoding alternates *predict*, where the dynamics carry the estimate forward, and *update*, where the new spikes correct it, weighted by the Kalman gain.

### Method 2: GRU

A gated recurrent network (128 hidden units) reads the square-rooted, z-scored counts bin by bin, causally, and outputs velocity. Unlike the Kalman filter, it can learn a nonlinear state, for example "the reach has started, and it's a curved one". It has far more parameters, so it uses dropout, weight decay and a best-validation checkpoint.

### Evaluation

MC_Maze Medium is a single session from one monkey, so we hold out *trials*:
- **NLB split:** the benchmark's own split, 188 training and 62 test trials.
- **Blocked CV:** 5 folds of *contiguous* trials, so a test trial's correlated neighbours can't be in training.

R² is the variance-weighted coefficient of determination over x and y velocity.

![Spike R²](figures/spikes-r2.png)

| | NLB split | 5-fold blocked CV |
|---|---|---|
| Kalman filter | **0.554** (CI 0.50–0.60) | **0.559** (CI 0.53–0.59) |
| GRU (3 seeds) | **0.746** (CI 0.61–0.83) | **0.800** (CI 0.76–0.81) |

Pairing the two models on the same test trials, the GRU is better by **+0.18 R²** on the NLB split (95% CI +0.05 to +0.28) and by **+0.23** under blocked CV (95% CI +0.20 to +0.26).

Integrating the decoded velocity reconstructs the reach paths (held-out trials):

![Trajectories](figures/spikes-trajectories.png)

![Velocity traces](figures/spikes-velocity-traces.png)

**What we learned**
- **The GRU wins clearly.** Its traces follow the bell-shaped speed profile and its reconstructed paths fan out in the right directions. The Kalman filter's paths are right on average but noisier and often too short.
- **How you set up the baseline matters as much as the model.** Our first Kalman filter, with a velocity-only state and 20 ms bins, scored R² 0.28. With 50 ms bins it reached 0.38, and adding position and acceleration to the state (the original Wu et al. 2003 formulation) gave 0.55. *Caveat:* I made these choices after seeing the NLB val-trial score of the first version. They follow the published formulations rather than a search, and the blocked-CV score (0.559) agrees, but the NLB-split Kalman number isn't fully untouched.
- **A real bug caught by a synthetic test.** The first Kalman update used the "information form", which inverts the state covariance. That breaks for states with zero variance, such as the constant term. A test on synthetic data with known tuning failed (R² −18), and the covariance-form update fixed it.

---

## 3. Spike sorting (synthetic demo)

MC_Maze arrives already sorted, so this demo shows the upstream step on a simulated 30 kHz recording with three neurons whose true spike times are known:

1. Filter 300–3000 Hz.
2. Detect crossings of −4.5× a robust noise estimate (median absolute value / 0.6745).
3. Cut aligned 1.6 ms waveforms.
4. Reduce them with PCA.
5. Cluster with a Gaussian mixture, with the number of units picked by BIC.

![Spike sorting](figures/spike-sorting-demo.png)

- **Results:** recall 98.8%, precision 100%, sorting accuracy 98.8%. On four more simulated recordings, sorting accuracy stays at 97–99%.
- **Over-splitting:** BIC over-splits on those recordings, finding 4–6 clusters for 3 neurons. In real sorters a merge step (or a person) fixes that.
- **Similar neurons merge:** in an earlier version, two neurons with similar filtered waveforms merged into one cluster, the opposite failure.

---

## Honest caveats

- **Offline decoding, not closed-loop control:** every number here is offline: decode recorded data, compare with what happened. In a live BCI the user sees the decoder's output and adapts to its errors, so offline R² or accuracy does not predict closed-loop performance well. That is why live systems are judged on task metrics such as bits per second, which this project does not measure.
- **Baselines, not state of the art:** this is a careful comparison of classic and standard neural decoders under honest splits. Stronger published models exist for both datasets; nothing here is meant to compete with them.
- **One session for spikes:** all spike results are within one session. Decoders usually degrade across days, which isn't measured here.
- **Small n for EEG statistics:** with 9 subjects, a Wilcoxon test can't go below p ≈ 0.004.
- **Compute caps:** the held-out-subject EEGNet folds were capped at 80 epochs and one seed, because training ran on a heavily loaded CPU.
- **Hand-written methods:** everything is implemented from scratch for clarity (no MNE, MOABB or pyRiemann). It's tested on synthetic data, but less battle-tested than those libraries.

## References

- Brunner et al. (2008), *BCI Competition 2008 – Graz data set A* (BCI IV 2a).
- Pei et al. (2021), *Neural Latents Benchmark '21*. Churchland et al. (2012), the maze task.
- Ramoser, Müller-Gerking & Pfurtscheller (2000), *Optimal spatial filtering of single trial EEG during imagined hand movement* (CSP).
- Lawhern et al. (2018), *EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces*.
- Wu et al. (2003), *Neural decoding of cursor motion using a Kalman filter*. Gilja et al. (2012), *A high-performance neural prosthesis enabled by control algorithm design*.
- Schlögl et al. (2007), *A fully automated correction method of EOG artifacts in EEG recordings*.
- Haufe et al. (2014), *On the interpretation of weight vectors of linear models in multivariate neuroimaging*.
- Quiroga, Nadasdy & Ben-Shaul (2004), *Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering*.
