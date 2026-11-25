# MUSE - Perceptual Focus

<p align="center">
  <strong>Make room for the sound that matters.</strong><br>
  <em>A real-time spectral-attention engine - not another equalizer.</em>
</p>

<p align="center">
  <img alt="Windows x64" src="https://img.shields.io/badge/platform-Windows%20x64-4479A1">
  <img alt="VST2" src="https://img.shields.io/badge/plugin-VST2-6A5ACD">
  <img alt="C++17" src="https://img.shields.io/badge/language-C%2B%2B17-00599C">
  <img alt="AVX1" src="https://img.shields.io/badge/SIMD-AVX1-orange">
  <img alt="Resolution" src="https://img.shields.io/badge/analysis-32%E2%80%93512%20regions-2E8B57">
</p>

> **The mix is not short of information. Sometimes too many parts are competing to be heard at once.**
>
> MUSE approaches that problem by estimating spectral masking relationships and gently attenuating likely maskers around regions that the listener has chosen to preserve. It does not try to make every frequency louder, brighter or more prominent. It tries to make the existing information easier to follow.

---

## The index

- [The opening frame](#the-opening-frame)
- [What MUSE is](#whatMUSE-es-is)
- [The question behind the engine](#the-question-behind-the-engine)
- [The project lineage](#the-project-lineage)
  - [CALP - content-aware loudness](#calp--content-aware-loudness)
  - [ASIP - local spatial opportunity](#asip--local-spatial-opportunity)
  - [DeFatiguer - perceptual stability over time](#defatiguer--perceptual-stability-over-time)
  - [SpectralGravity - macro spectral equilibrium](#spectralgravity--macro-spectral-equilibrium)
  - [What converged in MUSE](#what-converged-inMUSE)
- [What it does - and what it deliberately does not](#what-it-does--and-what-it-deliberately-does-not)
- [Inside the engine](#inside-the-engine)
  - [ERB analysis bank](#1-the-erb-analysis-bank)
  - [Bark-domain masking estimate](#2-the-bark-domain-masking-estimate)
  - [Preferred perceptual regions](#3-preferred-perceptual-regions)
  - [Masker attribution and subtractive gain](#4-masker-attribution-and-subtractive-gain)
  - [Stereo behavior](#5-stereo-behavior)
- [The controls](#the-controls)
- [The presets](#the-presets)
- [Time, tick rate, attack and release](#time-tick-rate-attack-and-release)
- [The parallel engine](#the-parallel-engine)
- [Reconstruction, response and signal integrity](#reconstruction-response-and-signal-integrity)
- [Compatibility and requirements](#compatibility-and-requirements)
- [Installation and use with Equalizer APO](#installation-and-use-with-equalizer-apo)
- [Build from source](#build-from-source)
- [Validation and field notes](#validation-and-field-notes)
- [Known limitations](#known-limitations)
- [A note about listening level](#a-note-about-listening-level)
- [Repository layout](#repository-layout)
- [Scientific context](#scientific-context)
- [License](#license)

---

## The opening frame

Imagine a dense mix as a scene with many actors speaking at the same time. The problem is not necessarily that one actor is too quiet, or that the entire scene needs more brightness. The problem may be that several voices occupy overlapping perceptual territory, making the information harder to follow.

A conventional equalizer asks which frequency should be raised or lowered. A compressor asks how much level should be controlled. A loudness processor asks how the programme should approach a target.

MUSE asks a different question:

> **Which spectral components are currently contributing to the masking of the regions I want to keep perceptually available - and how much of that competition can be reduced without redesigning the mix?**

That question defines the project.

MUSE is a Windows x64 VST2 processor intended for Equalizer APO and other compatible VST2 hosts. It is a real-time, stereo, psychoacoustic-inspired *self-masking* processor. It analyses the signal, estimates relationships between target regions and likely maskers, then allocates a smooth, bounded attenuation budget to the contributing regions.

The result is not intended to sound like a new mix. The intent is to make the existing mix feel less crowded, less insistent and easier to follow - while leaving its identity substantially intact.

## What MUSE is

**MUSE - Perceptual Focus** is a real-time perceptual spectral processor built around:

- a causal, ERB-spaced IIR analysis bank;
- five selectable analysis resolutions: **32, 64, 128, 256 and 512 regions**;
- Bark-domain spreading estimates for likely spectral maskers;
- user-defined priority regions: Body, Vocal, Presence, Articulation and Air;
- sigmoid-shaped masking decisions and attenuation allocation;
- per-band attack/release smoothing;
- independent left/right processing by default, with a continuous Stereo Link control;
- AVX1 acceleration with a scalar fallback;
- persistent Windows worker threads and a parallel decision path for high resolutions.

The project is deliberately not a general-purpose mastering chain. It has no loudness target, no conventional compressor threshold, no exciter, no stereo-width control and no source-separation stage.

### At a glance

| Property | Implementation |
|---|---|
| Project | MUSE - Perceptual Focus |
| Version | 1.3.0-dev |
| Format | Windows x64 VST2 |
| Host target | Equalizer APO; other compatible VST2 hosts may work |
| Processing | Stereo, 32-bit floating point |
| Analysis resolution | 32 / 64 / 128 / 256 / 512 ERB-spaced regions |
| Detector Update | 250–1500 Hz requested rate |
| Maximum per-region reduction | 0–9 dB |
| Stereo Link | 0–100% |
| SIMD | AVX1, runtime detected; scalar fallback |
| VST unique ID | `MESA` |
| VST version integer | `1300` |
| Parameters | 15 |

## The question behind the engine

MUSE grew out of a longer line of experiments around the same broad listening problem. The earlier projects approached different symptoms: loudness that ignored content, spectral energy that wandered, temporal and spatial instability, and listening fatigue.

MUSE is not a bundle of those processors chained together. It is a new engine built around a more specific central mechanism: **spectral competition and masking**.

Its guiding principles are:

1. **Analyse before intervening.** The signal is examined in perceptually motivated regions rather than treated as a single broadband level.
2. **Protect chosen information.** The user defines which broad perceptual regions deserve priority.
3. **Remove competition rather than manufacture presence.** The main intervention is attenuation of likely maskers, not positive EQ.
4. **Keep decisions continuous.** Sigmoid curves and time-based smoothing avoid hard threshold switching.
5. **Preserve channel independence unless asked otherwise.** The default detector is independent for L and R.
6. **Make the model inspectable.** The engine is deterministic DSP and explicit heuristics, not a black-box neural model.

## The project lineage

The four projects below are the conceptual family tree around MUSE. Each explored a different dimension of perceptual comfort and signal stability. Their influence is architectural and philosophical; **MUSE does not import their Python code, run them as dependencies, or reproduce their algorithms one-for-one**.

### CALP - content-aware loudness

[**CALP - Content-Aware Loudness Processor**](https://github.com/aston89/CALP-Content-Aware-Loudness-Processor)

CALP explored why a loudness processor should understand the material rather than blindly chase a number. Its analysis considers programme loudness, dynamics, crest factor, transient density and the distribution of energy across broad frequency regions. It uses smooth gain curves and multiband treatment to make loudness adjustment more content-aware.

**What CALP contributed to the thinking behind MUSE:**

- decisions should depend on what the signal contains, not only on raw level;
- different spectral regions should not automatically receive identical treatment;
- continuous gain curves are preferable to brittle on/off decisions;
- preserving the identity and proportions of the source matters more than applying a correction simply because a meter says so.

**What MUSE does differently:** CALP is an offline loudness-oriented workflow with a target-LUFS path and final peak protection. MUSE is a real-time masking processor. It does not chase LUFS, perform programme normalization or include CALP's limiter stage.

### ASIP - local spatial opportunity

[**ASIP - Adaptive Spatial Imaging Processor**](https://github.com/aston89/ASIP-Adaptive-Spatial-Imaging-Processor)

ASIP explored spatial enhancement as a local opportunity problem rather than a global widening command. It analyses Mid/Side information in the STFT domain and builds a time-frequency opportunity field from descriptors such as energy distribution, inter-channel coherence, phase dispersion, transient activity and Side entropy. The intended intervention is a constrained, locally generated spatial layer.

**What ASIP contributed to the thinking behind MUSE:**

- processing should happen only where the signal offers room for it;
- global transformations can be unnecessarily destructive;
- local evidence and smooth constraints can be more useful than a single global knob;
- stereo structure deserves explicit consideration rather than being treated as two unrelated mono files by default.

**What MUSE does differently:** MUSE is not a stereo imager. It does not synthesize a Side layer, widen the image or perform spatial diffusion. Its Stereo Link control blends independent and shared masking evidence; it is not an ASIP-style spatial opportunity field.

### DeFatiguer - perceptual stability over time

[**DeFatiguer - Adaptive Audio Processor**](https://github.com/aston89/DeFatiguer-Adaptive-Audio-Processor)

DeFatiguer approached listening effort as a problem of local instability. Its experimental pipeline uses perceptually grouped spectral information and combines spectral stability, temporal contrast, stereo/phase descriptors and micro-dynamic behaviour into a perceptual stability model. Its correction stage is designed to be conservative and to avoid obvious tonal redesign.

**What DeFatiguer contributed to the thinking behind MUSE:**

- listening comfort is not reducible to a static frequency-response curve;
- time behaviour matters, including how quickly spectral relationships change;
- a process can be useful even when the audible change is subtle;
- the goal can be to reduce unnecessary perceptual effort rather than to make the signal overtly brighter, louder or wider.

**What MUSE does differently:** MUSE does not calculate DeFatiguer's Artifact Stability Field, Temporal Contrast Field or a global fatigue score. It does not claim to measure or guarantee a reduction in human listening fatigue. Its mechanism is narrower: it estimates within-programme spectral masking and attenuates likely contributors.

### SpectralGravity - macro spectral equilibrium

[**SpectralGravity Processor**](https://github.com/aston89/SpectralGravity-Processor)

SpectralGravity explored broad energy stability across Low, Mid and High regions. Its purpose is not loudness maximization but gentle macro-level correction: observe how energy is distributed over time, then apply very small, linked adjustments while protecting stereo structure and tonal identity.

**What SpectralGravity contributed to the thinking behind MUSE:**

- spectral balance is dynamic, not merely a static EQ curve;
- the distribution of energy matters as much as the absolute amount;
- small, continuous interventions can be more transparent than large corrective moves;
- stability and equilibrium can be more valuable than obvious enhancement.

**What MUSE does differently:** SpectralGravity works at a coarse Low/Mid/High level. MUSE works across up to 512 ERB-spaced analysis regions and evaluates masking relationships in a Bark-mapped model. It does not apply SpectralGravity's three-band balancing process.

### What converged in MUSE

The lineage can be summarized as four complementary questions:

| Project | Primary question | Dimension explored |
|---|---|---|
| CALP | How should gain respond to the actual content? | Content-aware loudness and dynamics |
| ASIP | Where is spatial intervention locally admissible? | Spatial opportunity and stereo structure |
| DeFatiguer | Which instabilities make audio harder to follow over time? | Temporal, spectral and spatial stability |
| SpectralGravity | How does broad spectral energy move and settle? | Macro spectral equilibrium |
| **MUSE** | Which regions are masking the regions I want to preserve? | **Within-programme spectral competition** |

MUSE is the point where those explorations meet around one real-time decision engine. It does not attempt to solve every problem addressed by the earlier projects. It concentrates on one mechanism and tries to execute it at high resolution, with controlled timing and predictable gain behaviour.

## What it does - and what it deliberately does not

### MUSE does

- split incoming stereo audio into perceptually spaced analysis regions;
- estimate each region's evolving energy;
- estimate how energy in other regions may mask preferred target regions;
- determine which regions contribute to that masking estimate;
- allocate smooth, bounded attenuation to those likely maskers;
- preserve preferred regions according to user-defined priorities;
- maintain independent channel behaviour by default;
- let the user trade CPU load, spectral resolution and decision cadence.

### MUSE does not

- identify, isolate or separate a singer, instrument or sound source;
- recognize speech or understand musical arrangement;
- know which instrument is playing in a given region;
- create positive per-band EQ boosts;
- normalize integrated loudness or target LUFS;
- replace a true-peak limiter or a safety limiter;
- repair clipping, distortion, poor recording technique or a bad mix;
- widen the stereo image or synthesize new spatial content;
- model every form of simultaneous, temporal, binaural or level-dependent masking;
- predict the response of an individual listener, headphone, loudspeaker or room.

A preset named **Vocal Center** means that the region associated with vocal fundamentals/formants receives higher priority. It does **not** mean that MUSE detects or follows a voice.

## Inside the engine

The high-level processing path is:

```text
STEREO INPUT
    │
    ▼
ERB-SPACED IIR ANALYSIS BANK
    │  32 / 64 / 128 / 256 / 512 regions
    ▼
PER-REGION POWER ENVELOPES
    │  sample-by-sample attack / release
    ▼
BARK-DOMAIN SPREADING ESTIMATE
    │  target regions ← likely masker contributions
    ▼
PREFERRED-REGION WEIGHTING
    │  Body / Vocal / Presence / Articulation / Air
    ▼
MASKER ATTRIBUTION + SIGMOID DECISION
    │
    ▼
SUBTRACTIVE GAIN TARGETS
    │  bounded attenuation, smoothed over time
    ▼
PER-CHANNEL RECONSTRUCTION
    │
    ▼
STEREO OUTPUT
```

### 1. The ERB analysis bank

MUSE uses a cascade of second-order low-pass sections whose cutoff frequencies are distributed along an ERB-rate axis. Adjacent low-pass outputs are differenced to form complementary analysis regions:

- the first region is the lowest low-pass output;
- each middle region is the difference between two neighbouring low-pass outputs;
- the final region is the input minus the last low-pass output.

With all region gains set to unity, these complementary components reconstruct the input to floating-point precision. This is a causal IIR analysis bank; the regions are not generated by an FFT/STFT window.

The selectable modes are 32, 64, 128, 256 and 512 analysis regions. Their centre frequencies are mapped to Bark for the masking model.

**Important terminology:** 512 analysis regions do not mean 512 independent physiological critical bands. They are numerical analysis channels distributed on an ERB scale. More regions provide a finer partition for this particular engine; they do not imply 512 separate biological filters in the ear.

### 2. The Bark-domain masking estimate

The engine maps each analysis-region centre frequency to Bark and evaluates an asymmetric spreading function. The implemented spreading curve is:

\[
S(\Delta z)=15.81+7.5x-17.5\sqrt{1+x^2},\qquad x=\Delta z+0.474
\]

where \(\Delta z\) is the Bark-distance between a target region and a candidate masker. Contributions below **−55 dB** are omitted from the sparse tables.

For each preferred target region, MUSE estimates weighted energy arriving from other regions through that spreading function. It compares the target's own energy with the estimated masking energy, then passes the relationship through smooth logistic functions. This is a practical engineering approximation inspired by auditory-filter and masking concepts - not a complete physiological model of hearing.

### 3. Preferred perceptual regions

Five broad preference anchors are exposed to the user. Their centres are expressed in Bark:

| Control | Anchor centre | Intended emphasis |
|---|---:|---|
| Body | 4.8 Bark | Weight, body and lower-mid foundation |
| Vocal | 8.6 Bark | Vocal core and nearby formant region |
| Presence | 12.5 Bark | Presence and forward definition |
| Articulation | 16.3 Bark | Consonant/detail articulation |
| Air | 21.0 Bark | Upper-air region |

Each preference is shaped around its anchor with a Gaussian profile. **Focus Width** controls the width of those profiles. Overlapping preference regions combine into a continuous priority map instead of hard frequency borders.

These are spectral preference regions, not source labels. Their purpose is to tell the masking model what the user wants to keep perceptually prioritised.

### 4. Masker attribution and subtractive gain

After target regions have been weighted by preference, audibility and estimated maskedness, the engine attributes each target's masking contribution back to candidate masker regions. The contribution of a masker is normalized within the target's masking estimate so that the decision is based on relative contribution, not merely on the fact that a region contains energy.

A sigmoid gate turns the resulting evidence into a gain target. The attenuation budget is controlled by Focus Amount and Maximum Reduction, with the latter capped at **9 dB per analysis region**. Energy Preference protects regions that the user has marked as important.

The gain targets are smoothed sample by sample. The processor does not apply positive per-region gains: individual region gains are bounded at or below unity.

### 5. Stereo behavior

MUSE processes the left and right channels independently by default.

- **Stereo Link = 0%:** each channel uses its own masking evidence and gain decisions.
- **Stereo Link = 100%:** the detector uses shared stereo evidence and a common gain curve.
- **Intermediate values:** local and linked attenuation targets are blended in dB.

The feature controls how masking decisions are linked. It is not a stereo-width parameter and does not create a new Side signal.

## The controls

The plugin exposes 15 parameters. Factory defaults are conservative starting points; every preset remains editable.

| Control | Range / choices | Factory default | Role |
|---|---|---:|---|
| Focus Amount | 0–100% | 42% | Overall strength of the attenuation allocation |
| Mask Sensitivity | 0–100% | 58% | How readily masking evidence activates reduction |
| Sigmoid Softness | 0–100% | 62% | Smoothness of masking and gain transitions |
| Maximum Reduction | 0–9 dB | 3.5 dB | Upper limit for per-region attenuation |
| Energy Preference | 0–100% | 72% | Strength of protection for preferred regions |
| Body | 0–100% | 22% | Priority around the Body anchor |
| Vocal | 0–100% | 62% | Priority around the Vocal anchor |
| Presence | 0–100% | 42% | Priority around the Presence anchor |
| Articulation | 0–100% | 25% | Priority around the Articulation anchor |
| Air | 0–100% | 8% | Priority around the Air anchor |
| Focus Width | 0.35–4.0 Bark | 1.35 Bark | Width of the preference profiles |
| Responsiveness | 0–100% | 72% | Power-envelope and gain-curve attack/release behaviour |
| Band Resolution | 32 / 64 / 128 / 256 / 512 | 64 | Number of ERB analysis regions |
| Stereo Link | 0–100% | 0% | Blend between independent and linked decisions |
| Detector Update | 250–1500 Hz | 1000 Hz | Requested masking-decision refresh rate |

### How to think about the main controls

- **Focus Amount** is the overall intervention amount. It is not output volume.
- **Mask Sensitivity** changes how readily the masking model treats evidence as actionable.
- **Energy Preference** changes how strongly selected target regions are protected from attenuation.
- **Focus Width** determines whether a preference is concentrated or spread over a broader spectral neighbourhood.
- **Responsiveness** changes the temporal envelope and gain smoothing; it does not set how often the detector runs.
- **Detector Update** controls how often new masking targets are calculated; it does not replace attack/release smoothing.
- **Band Resolution** changes analysis granularity and CPU cost. More regions are not automatically better for every source.

## The presets

The preset names describe starting configurations, not automatic content recognition.

| Preset | Intended starting point | Resolution / detector |
|---|---|---:|
| Factory Default | Neutral, balanced starting point | 64 / 1000 Hz |
| Natural Balance | Gentle broad cleanup with strong energy preservation | 128 / 1250 Hz |
| Vocal Center | Preserve the vocal region and attenuate estimated maskers around it | 256 / 1250 Hz |
| Speech Intelligibility | Give more priority to formants and consonant detail | 256 / 1500 Hz |
| Presence + Air | Prioritise presence and upper articulation without a static high shelf | 256 / 1250 Hz |
| Warm Foundation | Prioritise body and lower-mid continuity | 128 / 1000 Hz |
| Dense Mix | Broad preference map for dense material | 512 / 750 Hz |
| Low CPU | Conservative reduced-cost configuration | 64 / 750 Hz |

A sensible way to explore is to begin with Natural Balance or Factory Default, then increase resolution before increasing Detector Update. The best setting depends on the host, block size, sample rate, processor and material.

## Time, tick rate, attack and release

MUSE contains three distinct time domains. They work together, but they are not interchangeable.

### Power-envelope timing

The per-region power envelope is updated on every audio sample. Responsiveness maps approximately between:

- attack: **7.0 ms → 1.5 ms**;
- release: **150 ms → 35 ms**.

These envelope constants are sample-rate based. They do not become faster merely because Detector Update is increased.

### Decision timing

Detector Update requests a target-decision cadence from 250 to 1500 Hz. The requested rate is converted into an integer sample stride:

```text
stride = round(sample_rate / requested_update_hz)
```

The stride is bounded to at least 32 samples and to a maximum corresponding to 250 Hz. Because the interval must be an integer number of samples, the effective rate is quantized.

Examples:

| Sample rate | Requested | Effective interval | Effective rate |
|---:|---:|---:|---:|
| 48 kHz | 1000 Hz | 48 samples | 1000 Hz |
| 48 kHz | 1500 Hz | 32 samples | 1500 Hz |
| 44.1 kHz | 1500 Hz | 32 samples | ≈1378 Hz |
| 192 kHz | 250 Hz | 768 samples | 250 Hz |

The GUI shows the requested value, while the engine uses the quantized interval.

### Gain smoothing

The gain attack cannot settle faster than the interval at which fresh targets are supplied. In simplified form:

```text
gain_attack_ms = max(1.5 ms,
                     0.8 × responsiveness_attack_ms,
                     decision_interval_ms)
```

Gain release remains time-based, with a minimum bound that also respects the decision interval. This avoids a gain curve racing ahead of stale decision data. It does not remove the sample-and-hold nature of the detector: a new change can wait up to one decision interval for the next target refresh.

## The parallel engine

MUSE uses a bounded, persistent worker architecture. It does not create a new thread for each sample, band or detector tick.

### Channel workers

Two persistent workers handle the sample-recursive channel paths:

- one worker owns the left-channel filter/envelope/gain/synthesis state;
- one worker owns the right-channel state.

The filter bank itself is recursive across time for a given channel, so consecutive samples from that channel remain in order.

### Decision workers

On systems exposing at least four logical processors, the plugin attempts to create four additional persistent decision workers - two associated with the left-channel worker and two with the right-channel worker. They are used for the 256- and 512-region decision calculations. At lower resolutions the decision work remains serial to avoid unnecessary task overhead.

The decision stage is split into independent ranges:

1. target-side statistics are calculated for disjoint target ranges;
2. masker-side evidence and gain assignment are calculated for disjoint masker ranges.

A transposed CSR spreading table makes the second phase masker-centric: each worker owns its own masker range and writes to disjoint evidence/gain positions. No atomic accumulation into shared masker cells is required.

### Block scheduling

At **Stereo Link = 0%**, the coordinator prepares a bounded control-snapshot schedule for a block segment, submits one whole-channel job to each channel worker, then waits for both. The detector still refreshes at the scheduled tick positions; it is not reduced to one decision per host block. A segment is bounded to at most 32,768 samples.

This removes the old host-callback wake/wait cycle at every decision tick. The decision workers still have internal phase barriers because the masker phase consumes the target phase's results.

At **Stereo Link above 0%**, v1.3 retains the coordinated linked-detector path rather than changing the timing semantics of the shared evidence calculation. Local decision work can still use the decision workers at 256/512 regions, but the shared linked path remains more tightly coordinated.

### SIMD

The optional AVX1 kernels process eight 32-bit floating-point lanes per 256-bit vector for the filter-bank and power-envelope operations. Runtime dispatch checks CPU AVX support and the operating-system state required to use AVX safely.

There is deliberately no AVX2, FMA or AVX-512 requirement. A baseline scalar x64 path remains available when AVX1 is not usable. The project targets useful parallel scheduling and AVX1 on a wide range of CPUs rather than requiring a newer instruction set.

## Reconstruction, response and signal integrity

The analysis bank is complementary: when all region gains are unity, the input is reconstructed to floating-point precision. With different region gains, the output is the complex sum of overlapping filter responses.

That distinction matters. Individual per-region gains are never greater than unity, but this fact alone does **not** prove that every possible combination of overlapping complex responses has a total magnitude at or below 0 dB at every frequency.

The regression suite checks the complex transfer function across 4096 frequency points for engine-generated profiles at all five resolutions, plus eight deliberately varied 512-region profiles (alternating, notched, stepped and smooth). The selected synthetic profiles are constrained to a peak no greater than 1.02 linear, approximately +0.17 dB.

This is a regression envelope for the tested profiles - **not** a universal limiter, not a proof for all arbitrary gain profiles and not a guarantee about every time-varying transient. No global output attenuation was added because a generic worst-case bound would have reduced level even when the measured profile was already below unity.

A resolution change resets the relevant filter state and crossfades back in over approximately 20 ms. This is a transition, not 20 ms of inserted audio latency. The VST reports zero plugin delay and uses a causal IIR processing path without look-ahead.

## Compatibility and requirements

- Windows x64.
- VST2-compatible host.
- Stereo float input/output.
- C++17 source build.
- AVX1 is optional; the scalar fallback is retained.
- Equalizer APO is the primary target host used during development.

The plugin does not expose a double-precision VST processing callback. The build workflow fetches the VST2 interface headers from [Xaymar/vst2sdk](https://github.com/Xaymar/vst2sdk) at tag `v0.4.0`; those headers are fetched during the build and are not copied into this repository.

## Installation and use with Equalizer APO

1. Build the repository through GitHub Actions, or use a compatible Windows x64 build of the plugin.
2. Download the workflow artifact containing `MUSE_Perceptual_Focus.dll`.
3. Place the DLL in the VST2 location used by your Equalizer APO VST host/configuration.
4. Rescan or restart the host so it loads the current DLL.
5. Start with Factory Default or Natural Balance. Set Stereo Link to 0% unless linked behaviour is specifically desired.
6. Increase resolution and Detector Update gradually while monitoring real-time CPU load and listening for dropouts.

A practical high-resolution starting point is 256 regions at 1000–1250 Hz. 512 regions are available as an experimental maximum-resolution mode. The Dense Mix preset intentionally starts at 512 / 750 Hz; on a capable system, higher detector settings can be explored separately.

## Build from source

The repository intentionally stays small:

```text
.github/workflows/build.yml
muse_perceptual.cpp
README.md
```

GitHub Actions performs the Windows x64 build:

1. checks out the repository;
2. configures MSVC for x64;
3. fetches the VST2 interface headers;
4. compiles the optional AVX1 kernel object;
5. compiles and runs the DSP regression suite;
6. builds the VST2 DLL;
7. verifies PE32+ AMD64 and the `VSTPluginMain` export;
8. uploads `MUSE_Perceptual_Focus.dll` as an artifact.

All MSVC compile steps explicitly use `/utf-8`. Preset menu em dashes are represented by C++ universal character escapes (`\u2014`) so the labels do not depend on the Windows machine's active ANSI code page.

To run the same core regression suite locally from an x64 Native Tools Command Prompt for Visual Studio, the essential commands are:

```powershell
New-Item -ItemType Directory -Force build | Out-Null

cl /nologo /std:c++17 /O2 /W4 /permissive- /EHsc /utf-8 `
  /c /DMUSE_AVX_KERNEL /arch:AVX muse_perceptual.cpp `
  /Fo:build/muse_avx.obj

cl /nologo /std:c++17 /O2 /W4 /permissive- /EHsc /utf-8 `
  /DMUSE_SELF_TEST muse_perceptual.cpp build/muse_avx.obj `
  /Fe:build/muse_tests.exe

.\build\muse_tests.exe
```

The plugin DLL build additionally needs the fetched `vst.h` include directory and the Windows libraries shown in `.github/workflows/build.yml`.

## Validation and field notes

The v1.3 regression suite contains **85 checks**. The documented local development validation is:

| Validation | Result |
|---|---:|
| GCC optimized DSP suite | 85/85 passed |
| Clang optimized DSP suite | 85/85 passed |
| AddressSanitizer + UndefinedBehaviorSanitizer | 85/85 passed |
| ThreadSanitizer | 85/85 passed |

The suite covers the five resolutions, unity reconstruction, CSR and transposed-CSR integrity, detector-rate quantization, attack/release timing, block/sample consistency, irregular host block sizes, serial/parallel decision parity, Stereo Link behaviour and selected complex-response regression profiles.

**Maintainer field report:** MUSE v1.3 has been reported running in Equalizer APO at 512 regions and a requested 1500 Hz Detector Update without dropouts, with processing activity observed across four CPU cores. This is a result on one target system, not a universal performance guarantee or a published benchmark. Actual CPU use and scheduling depend on the host, block size, processor topology, background load and sample rate.

The Windows MSVC build and live Win32 worker lifecycle should be verified by GitHub Actions for each clean repository build. The in-process GCC/Clang test harness validates the DSP and parallel decision calculations; it does not replace a live Windows host test.

## Known limitations

- This is a single-program self-masking approximation, not a source separator.
- ERB regions are analysis channels, not independent physiological critical bands.
- Bark spreading, target weighting, masker attribution and gain allocation are psychoacoustically motivated engineering heuristics, not a direct implementation of one published auditory model.
- The model does not reproduce all temporal, binaural, level-dependent or listener-specific masking effects.
- No global LUFS target, true-peak limiter, loudness normalizer or final output safety limiter is included.
- Per-region attenuation does not mathematically guarantee a total response at or below 0 dB for every possible gain profile; see [Reconstruction, response and signal integrity](#reconstruction-response-and-signal-integrity).
- 512 regions are computationally expensive and are not automatically the best choice for every source or host.
- Stereo Link above 0% retains the coordinated path; the block-batched independent path is most efficient at 0%.
- A requested Detector Update rate is quantized to integer sample intervals.
- A smoother subjective result is not proof that a signal is objectively safer, quieter or free from hearing risk.

## A note about listening level

MUSE is designed to make some forms of spectral competition less intrusive. A smoother presentation can make it tempting to turn the playback level up. MUSE does not measure sound pressure at the listening position and is **not** a hearing-protection device, loudness monitor or output limiter. Keep playback levels and exposure time under control independently of the plugin.

## Repository layout

```text
MUSE-ES-Perceptual-Focus/
├── .github/
│   └── workflows/
│       └── build.yml
├── muse_perceptual.cpp
└── README.md
```

The C++ source contains the baseline DSP/VST2 implementation, the optional AVX1 kernel and the regression-test harness, selected by compile-time definitions. The workflow builds the AVX1 object separately and links it into the Windows x64 VST2 DLL.

## Scientific context

MUSE uses established psychoacoustic ideas as foundations, then adds its own engineering heuristics for target preference, masker attribution and gain allocation. The references below support the auditory-filter and masking context; they should not be read as claiming that MUSE is a faithful implementation of either paper.

1. **Glasberg, B. R. & Moore, B. C. J. (1990).** “Derivation of auditory filter shapes from notched-noise data.” *Hearing Research*, 47(1–2), 103–138. [doi:10.1016/0378-5955(90)90170-T](https://doi.org/10.1016/0378-5955(90)90170-T). Experimental derivation of auditory-filter shapes and frequency-dependent selectivity.
2. **Schroeder, M. R., Atal, B. S. & Hall, J. L. (1979).** “Optimizing digital speech coders by exploiting masking properties of the human ear.” *Journal of the Acoustical Society of America*, 66, 1647–1652. [doi:10.1121/1.383662](https://doi.org/10.1121/1.383662). A foundational example of using auditory masking properties in signal-processing decisions.

The ERB-spaced filter bank and Bark-domain spreading are psychoacoustically motivated. The exact target curves, preference combination, normalized masker attribution and attenuation budget are MUSE-specific design choices.

---

<p align="center">
  <strong>MUSE</strong><br>
  <em>Not every sound needs to be louder. Some sounds need a little more room.</em>
</p>
