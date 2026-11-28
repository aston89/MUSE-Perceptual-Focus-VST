/*
 * MUSE Perceptual Focus v1.3.3 (FTZ/DAZ denormal protection, AVX1, block-batched stereo workers, parallel CSR decisions, unified worker priority)
 * Real-time, subtractive, preference-guided self-masking processor.
 * Build the baseline source normally. Compile this same file once more with
 * /DMUSE_AVX_KERNEL /arch:AVX to produce the optional AVX1 kernel object.
 */

#if defined(MUSE_AVX_KERNEL)
#include <algorithm>
#include <cmath>
#include <immintrin.h>

extern "C" void muse_avx_make_bands(
    int splitCount, int bandCount, float input,
    const float* b0, const float* b1, const float* b2,
    const float* a1, const float* a2,
    float* z1, float* z2, float* low, float* band) noexcept {
    const __m256 x = _mm256_set1_ps(input);
    int i = 0;
    for (; i + 8 <= splitCount; i += 8) {
        const __m256 y = _mm256_add_ps(
            _mm256_mul_ps(_mm256_loadu_ps(b0 + i), x),
            _mm256_loadu_ps(z1 + i));
        const __m256 nz1 = _mm256_add_ps(
            _mm256_sub_ps(_mm256_mul_ps(_mm256_loadu_ps(b1 + i), x),
                          _mm256_mul_ps(_mm256_loadu_ps(a1 + i), y)),
            _mm256_loadu_ps(z2 + i));
        const __m256 nz2 = _mm256_sub_ps(
            _mm256_mul_ps(_mm256_loadu_ps(b2 + i), x),
            _mm256_mul_ps(_mm256_loadu_ps(a2 + i), y));
        _mm256_storeu_ps(low + i, y);
        _mm256_storeu_ps(z1 + i, nz1);
        _mm256_storeu_ps(z2 + i, nz2);
    }
    for (; i < splitCount; ++i) {
        const float y = b0[i] * input + z1[i];
        const float nz1 = b1[i] * input - a1[i] * y + z2[i];
        const float nz2 = b2[i] * input - a2[i] * y;
        low[i] = y; z1[i] = nz1; z2[i] = nz2;
    }
    band[0] = low[0];
    for (int k = 1; k < splitCount; ++k) band[k] = low[k] - low[k - 1];
    band[bandCount - 1] = input - low[splitCount - 1];
}

extern "C" void muse_avx_update_powers(
    int bandCount, const float* bands, float* power,
    float attackAlpha, float releaseAlpha) noexcept {
    const __m256 attack = _mm256_set1_ps(attackAlpha);
    const __m256 release = _mm256_set1_ps(releaseAlpha);
    const __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= bandCount; i += 8) {
        const __m256 b = _mm256_loadu_ps(bands + i);
        const __m256 oldPower = _mm256_loadu_ps(power + i);
        const __m256 p = _mm256_mul_ps(b, b);
        const __m256 mask = _mm256_cmp_ps(p, oldPower, _CMP_GT_OQ);
        const __m256 alpha = _mm256_blendv_ps(release, attack, mask);
        __m256 next = _mm256_add_ps(oldPower,
            _mm256_mul_ps(alpha, _mm256_sub_ps(p, oldPower)));
        const __m256 ordered = _mm256_cmp_ps(next, next, _CMP_ORD_Q);
        next = _mm256_and_ps(next, ordered);
        next = _mm256_max_ps(next, zero);
        _mm256_storeu_ps(power + i, next);
    }
    for (; i < bandCount; ++i) {
        const float p = bands[i] * bands[i];
        const float alpha = p > power[i] ? attackAlpha : releaseAlpha;
        const float next = power[i] + alpha * (p - power[i]);
        power[i] = std::isfinite(next) && next >= 0.0f ? next : 0.0f;
    }
}

#else // baseline DSP / VST2 / self tests

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cwchar>
#include <cstring>
#include <limits>
#include <new>
#include <vector>
#include <complex>
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#include <xmmintrin.h>
#define MUSE_HAS_MXCSR 1
#else
#define MUSE_HAS_MXCSR 0
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__i386__) || defined(__x86_64__)
#include <cpuid.h>
#endif
#endif

extern "C" void muse_avx_make_bands(
    int splitCount, int bandCount, float input,
    const float* b0, const float* b1, const float* b2,
    const float* a1, const float* a2,
    float* z1, float* z2, float* low, float* band) noexcept;
extern "C" void muse_avx_update_powers(
    int bandCount, const float* bands, float* power,
    float attackAlpha, float releaseAlpha) noexcept;

namespace muse_perceptual {

#if MUSE_HAS_MXCSR
constexpr unsigned int kMxcsrDazMask = 1u << 6;
constexpr unsigned int kMxcsrFtzMask = 1u << 15;
#endif

// Flush denormal inputs/results while this thread performs audio DSP. MXCSR is
// per-thread, so every persistent worker must establish this mode independently.
// The scoped form is used on host-owned callback threads and restores their
// incoming floating-point environment before returning to the host.
class ScopedDenormalMode final {
public:
    ScopedDenormalMode() noexcept {
#if MUSE_HAS_MXCSR
        previousMxcsr_ = _mm_getcsr();
        _mm_setcsr(previousMxcsr_ | kMxcsrDazMask | kMxcsrFtzMask);
        active_ = true;
#endif
    }
    ~ScopedDenormalMode() noexcept {
#if MUSE_HAS_MXCSR
        if (active_) _mm_setcsr(previousMxcsr_);
#endif
    }
    ScopedDenormalMode(const ScopedDenormalMode&) = delete;
    ScopedDenormalMode& operator=(const ScopedDenormalMode&) = delete;
private:
#if MUSE_HAS_MXCSR
    unsigned int previousMxcsr_ = 0;
    bool active_ = false;
#endif
};

inline bool cpuSupportsAvx() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int regs[4]{};
    __cpuid(regs, 1);
    constexpr int kOsXsave = 1 << 27;
    constexpr int kAvx = 1 << 28;
    if ((regs[2] & (kOsXsave | kAvx)) != (kOsXsave | kAvx)) return false;
    return (_xgetbv(0) & 0x6) == 0x6;
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__i386__) || defined(__x86_64__))
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
    constexpr unsigned int kOsXsave = 1u << 27;
    constexpr unsigned int kAvx = 1u << 28;
    if ((ecx & (kOsXsave | kAvx)) != (kOsXsave | kAvx)) return false;
    unsigned int xcrLow = 0, xcrHigh = 0;
    __asm__ volatile("xgetbv" : "=a"(xcrLow), "=d"(xcrHigh) : "c"(0));
    (void)xcrHigh;
    return (xcrLow & 0x6u) == 0x6u;
#else
    return false;
#endif
}

constexpr std::size_t kMaxBands = 512;
constexpr std::size_t kMaxSplits = kMaxBands - 1;
constexpr std::size_t kResolutionCount = 5;
constexpr std::size_t kMaxScheduledBlockSamples = 32768;
constexpr std::size_t kMaxControlSnapshots = kMaxScheduledBlockSamples / 32 + 2;
constexpr std::array<int, kResolutionCount> kBandChoices = {32, 64, 128, 256, 512};
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSpreadFloorDb = -55.0;

inline double clamp(double x, double lo, double hi) noexcept {
    if (!std::isfinite(x)) return lo;
    return std::max(lo, std::min(hi, x));
}
inline float clampf(float x, float lo, float hi) noexcept {
    if (!std::isfinite(x)) return lo;
    return std::max(lo, std::min(hi, x));
}
inline double logistic(double x) noexcept {
    x = clamp(x, -40.0, 40.0);
    return 1.0 / (1.0 + std::exp(-x));
}
inline double sigmoidUnit(double x) noexcept {
    x = clamp(x, 0.0, 1.0);
    constexpr double k = 6.0;
    const double lo = logistic(-0.5 * k);
    const double hi = logistic(0.5 * k);
    return clamp((logistic(k * (x - 0.5)) - lo) / (hi - lo), 0.0, 1.0);
}
inline double dbToLinear(double db) noexcept { return std::pow(10.0, db / 20.0); }
inline double powerToDb(double p) noexcept { return 10.0 * std::log10(std::max(p, 1.0e-24)); }
inline double timeAlpha(double sampleRate, double milliseconds) noexcept {
    const double seconds = clamp(milliseconds, 0.1, 2000.0) * 0.001;
    return 1.0 - std::exp(-1.0 / (seconds * sampleRate));
}
inline double erbRate(double frequencyHz) noexcept {
    const double f = std::max(0.0, frequencyHz) / 1000.0;
    return 21.4 * std::log10(1.0 + 4.37 * f);
}
inline double inverseErbRate(double rate) noexcept {
    return (std::pow(10.0, rate / 21.4) - 1.0) * 1000.0 / 4.37;
}
inline double hzToBark(double frequencyHz) noexcept {
    const double f = std::max(0.0, frequencyHz);
    return 13.0 * std::atan(0.00076 * f) +
           3.5 * std::atan((f / 7500.0) * (f / 7500.0));
}
inline double barkSpreadingDb(double deltaBark) noexcept {
    const double shifted = deltaBark + 0.474;
    return 15.81 + 7.5 * shifted - 17.5 * std::sqrt(1.0 + shifted * shifted);
}

inline int normalizedToResolution(float v) noexcept {
    v = clampf(v, 0.0f, 1.0f);
    if (v < 0.125f) return 0;
    if (v < 0.375f) return 1;
    if (v < 0.625f) return 2;
    if (v < 0.875f) return 3;
    return 4;
}
inline float resolutionToNormalized(int mode) noexcept {
    mode = std::max(0, std::min(4, mode));
    return static_cast<float>(mode) * 0.25f;
}
inline int resolutionBandCount(int mode) noexcept {
    mode = std::max(0, std::min(4, mode));
    return kBandChoices[static_cast<std::size_t>(mode)];
}
inline int resolutionModeFromBandCount(double requestedBands) noexcept {
    int bestMode = 0;
    double bestDistance = std::numeric_limits<double>::max();
    for (int mode = 0; mode < static_cast<int>(kResolutionCount); ++mode) {
        const double distance = std::abs(requestedBands -
            static_cast<double>(kBandChoices[static_cast<std::size_t>(mode)]));
        if (distance < bestDistance) { bestDistance = distance; bestMode = mode; }
    }
    return bestMode;
}

struct Parameters {
    double amount = 0.42;
    double sensitivity = 0.58;
    double softness = 0.62;
    double maximumReductionDb = 3.5;
    double energyPreference = 0.72;
    double body = 0.22;
    double vocal = 0.62;
    double presence = 0.42;
    double articulation = 0.25;
    double air = 0.08;
    double focusWidthBark = 1.35;
    double responsiveness = 0.72;
    int resolutionMode = 1; // 32 / 64 / 128 / 256 / 512; default 64.
    double stereoLink = 0.0; // 0 = fully independent, 1 = fully linked.
    double detectorRateHz = 1000.0; // nominal decision refresh, independent of band count.
};

struct FilterCoefficients {
    std::array<float, kMaxSplits> b0{};
    std::array<float, kMaxSplits> b1{};
    std::array<float, kMaxSplits> b2{};
    std::array<float, kMaxSplits> a1{};
    std::array<float, kMaxSplits> a2{};
};

struct BandConfiguration {
    int bandCount = 64;
    int splitCount = 63;
    FilterCoefficients filters{};
    std::array<double, kMaxBands> centerHz{};
    std::array<double, kMaxBands> centerBark{};
    // Compressed Sparse Row (CSR): offsets delimit each target band's list.
    // The compact vectors are built only during prepare(), never in audio.
    std::array<uint32_t, kMaxBands + 1> spreadOffsets{};
    std::vector<uint16_t> spreadIndices{};
    std::vector<float> spreadWeights{};
    // Transposed CSR: each masker owns a disjoint list of contributing targets.
    std::array<uint32_t, kMaxBands + 1> transposeOffsets{};
    std::vector<uint16_t> transposeTargets{};
    std::vector<float> transposeWeights{};
};

struct ChannelState {
    // Structure-of-arrays: independent biquad states are SIMD-friendly.
    std::array<float, kMaxSplits> z1{};
    std::array<float, kMaxSplits> z2{};
    std::array<float, kMaxSplits> low{};
    std::array<float, kMaxBands> power{};
    std::array<float, kMaxBands> band{};
};

class StereoEngine {
public:
    static constexpr std::size_t kMaxParallelChunk = 256;
    struct ControlSnapshot {
        std::size_t offset = 0;
        Parameters parameters{};
        float powerAttackAlpha = 0.0f;
        float powerReleaseAlpha = 0.0f;
        float gainAttackAlpha = 0.0f;
        float gainReleaseAlpha = 0.0f;
    };
    struct DecisionScratch {
        std::array<double, kMaxBands> powers{};
        std::array<double, kMaxBands> targetWeights{};
        std::array<double, kMaxBands> maskTotals{};
        std::array<double, kMaxBands> evidence{};
        std::array<double, kMaxBands> gainDb{};
        std::array<double, 2> partialNormalization{};
        double normalization = 0.0;
    };
    struct ChannelControl {
        Parameters current{};
        std::array<double, kMaxBands> preferenceCache{};
        float powerAttackAlpha = 0.0f;
        float powerReleaseAlpha = 0.0f;
        float gainAttackAlpha = 0.0f;
        float gainReleaseAlpha = 0.0f;
        bool preferenceInitialized = false;
    };
    using ChannelSubmit = void(*)(void*, StereoEngine*, int, const float*, float*,
                                  std::size_t, bool, bool, bool) noexcept;
    using ChannelsWait = void(*)(void*) noexcept;
    using DecisionParallel = void(*)(void*, StereoEngine*, int, int, int) noexcept;

    explicit StereoEngine(bool enableAvx = true) noexcept : avxEnabled_(enableAvx && cpuSupportsAvx()) {}

    bool prepare(double sampleRate) noexcept {
        if (!std::isfinite(sampleRate) || sampleRate < 8000.0 || sampleRate > 768000.0)
            return false;
        sampleRate_ = sampleRate;
        for (std::size_t i = 0; i < kResolutionCount; ++i) {
            if (!buildConfiguration(configurations_[i], sampleRate, kBandChoices[i])) {
                prepared_ = false;
                return false;
            }
        }
        activeMode_ = current_.resolutionMode;
        activeConfig_ = &configurations_[static_cast<std::size_t>(activeMode_)];
        reset();
        updateControlRate();
        smoothParameters();
        rebuildPreferenceCache();
        updateTimeConstants();
        syncChannelControlsFromGlobal();
        prepared_ = true;
        return true;
    }

    void setParameters(const Parameters& p) noexcept {
        target_.amount = clamp(p.amount, 0.0, 1.0);
        target_.sensitivity = clamp(p.sensitivity, 0.0, 1.0);
        target_.softness = clamp(p.softness, 0.0, 1.0);
        target_.maximumReductionDb = clamp(p.maximumReductionDb, 0.0, 9.0);
        target_.energyPreference = clamp(p.energyPreference, 0.0, 1.0);
        target_.body = clamp(p.body, 0.0, 1.0);
        target_.vocal = clamp(p.vocal, 0.0, 1.0);
        target_.presence = clamp(p.presence, 0.0, 1.0);
        target_.articulation = clamp(p.articulation, 0.0, 1.0);
        target_.air = clamp(p.air, 0.0, 1.0);
        target_.focusWidthBark = clamp(p.focusWidthBark, 0.35, 4.0);
        target_.responsiveness = clamp(p.responsiveness, 0.0, 1.0);
        target_.resolutionMode = std::max(0, std::min(4, p.resolutionMode));
        target_.stereoLink = clamp(p.stereoLink, 0.0, 1.0);
        target_.detectorRateHz = clamp(p.detectorRateHz, 250.0, 1500.0);
        if (!parametersInitialized_) {
            current_ = target_;
            parametersInitialized_ = true;
            if (prepared_) {
                updateControlRate();
                updateTimeConstants();
                rebuildPreferenceCache();
                syncChannelControlsFromGlobal();
            }
        }
    }

    void reset() noexcept {
        resetChannel(left_);
        resetChannel(right_);
        for (auto& gains : targetGain_) gains.fill(1.0f);
        for (auto& gains : currentGain_) gains.fill(1.0f);
        linkedTargetDb_.fill(0.0);
        controlCounter_ = 0;
        preferenceCache_.fill(0.0);
        transitionSamplesRemaining_.fill(0);
        for (auto& control : channelControls_) {
            control.current = current_;
            control.preferenceCache.fill(0.0);
            control.preferenceInitialized = false;
            control.powerAttackAlpha = powerAttackAlpha_;
            control.powerReleaseAlpha = powerReleaseAlpha_;
            control.gainAttackAlpha = gainAttackAlpha_;
            control.gainReleaseAlpha = gainReleaseAlpha_;
        }
    }

    void processStereo(float inputL, float inputR, float& outputL, float& outputR) noexcept {
        if (!std::isfinite(inputL)) inputL = 0.0f;
        if (!std::isfinite(inputR)) inputR = 0.0f;
        if (!prepared_ || activeConfig_ == nullptr) {
            outputL = inputL; outputR = inputR; return;
        }
        if (target_.resolutionMode != activeMode_)
            activateResolution(target_.resolutionMode);
        if (current_.stereoLink > 1.0e-6 || target_.stereoLink > 1.0e-6) {
            if (controlCounter_ == 0) {
                refreshControlState();
                refreshChannelTargets(0, nullptr, nullptr);
                refreshChannelTargets(1, nullptr, nullptr);
            }
            outputL = processChannelSample(0, inputL);
            outputR = processChannelSample(1, inputR);
            if (++controlCounter_ >= controlStride_) controlCounter_ = 0;
        } else {
            // Match the block path's per-channel control timeline in the scalar API.
            if (controlCounter_ == 0) {
                refreshControlParameters();
                syncChannelControlFromSnapshot(0, current_, powerAttackAlpha_, powerReleaseAlpha_,
                                               gainAttackAlpha_, gainReleaseAlpha_);
                syncChannelControlFromSnapshot(1, current_, powerAttackAlpha_, powerReleaseAlpha_,
                                               gainAttackAlpha_, gainReleaseAlpha_);
                refreshChannelTargets(0, nullptr, nullptr);
                refreshChannelTargets(1, nullptr, nullptr);
            }
            outputL = processChannelSample(0, inputL);
            outputR = processChannelSample(1, inputR);
            if (++controlCounter_ >= controlStride_) controlCounter_ = 0;
        }
    }

    void processStereoBlock(const float* inputL, const float* inputR,
                            float* outputL, float* outputR, std::size_t samples,
                            bool accumulate, ChannelSubmit submitChannel = nullptr,
                            ChannelsWait waitChannels = nullptr, void* workerContext = nullptr) noexcept {
        if (outputL == nullptr || outputR == nullptr) return;
        if (!prepared_ || activeConfig_ == nullptr) {
            for (std::size_t i = 0; i < samples; ++i) {
                const float l = inputL != nullptr ? inputL[i] : 0.0f;
                const float r = inputR != nullptr ? inputR[i] : 0.0f;
                if (accumulate) { outputL[i] += l; outputR[i] += r; }
                else { outputL[i] = l; outputR[i] = r; }
            }
            return;
        }
        if (target_.resolutionMode != activeMode_)
            activateResolution(target_.resolutionMode);

        const bool independentMode = current_.stereoLink <= 1.0e-6 &&
                                     target_.stereoLink <= 1.0e-6;
        if (independentMode) {
            std::size_t offset = 0;
            while (offset < samples) {
                const std::size_t count = std::min(samples - offset, kMaxScheduledBlockSamples);
                buildControlSchedule(count);
                const float* lInput = inputL != nullptr ? inputL + offset : nullptr;
                const float* rInput = inputR != nullptr ? inputR + offset : nullptr;
                const bool useWorkers = submitChannel != nullptr && waitChannels != nullptr;
                if (useWorkers) {
                    submitChannel(workerContext, this, 0, lInput, outputL + offset,
                                  count, accumulate, false, true);
                    submitChannel(workerContext, this, 1, rInput, outputR + offset,
                                  count, accumulate, false, true);
                    waitChannels(workerContext);
                } else {
                    processChannelBlock(0, lInput, outputL + offset, count,
                                        accumulate, false, nullptr, nullptr, true);
                    processChannelBlock(1, rInput, outputR + offset, count,
                                        accumulate, false, nullptr, nullptr, true);
                }
                offset += count;
            }
            return;
        }

        std::size_t offset = 0;
        while (offset < samples) {
            const bool refreshTargets = controlCounter_ == 0;
            if (refreshTargets) refreshControlState();
            const std::size_t toControl = controlStride_ > controlCounter_
                ? controlStride_ - controlCounter_ : 1;
            const std::size_t count = std::min({samples - offset, toControl, kMaxParallelChunk});
            const float* lInput = inputL != nullptr ? inputL + offset : nullptr;
            const float* rInput = inputR != nullptr ? inputR + offset : nullptr;
            const bool useWorkers = submitChannel != nullptr && waitChannels != nullptr;
            if (useWorkers) {
                submitChannel(workerContext, this, 0, lInput, outputL + offset,
                              count, accumulate, refreshTargets, false);
                submitChannel(workerContext, this, 1, rInput, outputR + offset,
                              count, accumulate, refreshTargets, false);
                waitChannels(workerContext);
            } else {
                processChannelBlock(0, lInput, outputL + offset, count, accumulate, refreshTargets);
                processChannelBlock(1, rInput, outputR + offset, count, accumulate, refreshTargets);
            }
            controlCounter_ += count;
            if (controlCounter_ >= controlStride_) controlCounter_ = 0;
            offset += count;
        }
    }

    // Called only by the matching persistent channel worker, or by the
    // synchronous fallback when the host machine cannot start both workers.
    void processChannelBlock(int channel, const float* input, float* output,
                             std::size_t samples, bool accumulate,
                             bool refreshTargets,
                             DecisionParallel decisionParallel = nullptr,
                             void* decisionContext = nullptr,
                             bool scheduledIndependent = false) noexcept {
        if (scheduledIndependent) {
            processIndependentChannelBlock(channel, input, output, samples, accumulate,
                                           decisionParallel, decisionContext);
            return;
        }
        if (channel < 0 || channel > 1 || output == nullptr) return;
        ChannelState& state = channel == 0 ? left_ : right_;
        if (refreshTargets) refreshChannelTargets(channel, decisionParallel, decisionContext);
        for (std::size_t n = 0; n < samples; ++n) {
            float x = input != nullptr ? input[n] : 0.0f;
            if (!std::isfinite(x)) x = 0.0f;
            makeBands(state, x, *activeConfig_);
            updateBandPowers(state, channel, *activeConfig_);
            updateCurrentGains(channel);
            const float wet = synthesize(channel, state.band.data());
            const float blend = nextTransitionBlend(channel);
            float y = x + blend * (wet - x);
            if (!std::isfinite(y)) y = x;
            if (accumulate) output[n] += y;
            else output[n] = y;
        }
    }

    void processIndependentChannelBlock(int channel, const float* input, float* output,
                                        std::size_t samples, bool accumulate,
                                        DecisionParallel decisionParallel,
                                        void* decisionContext) noexcept {
        if (channel < 0 || channel > 1 || output == nullptr) return;
        ChannelState& state = channel == 0 ? left_ : right_;
        std::size_t snapshotIndex = 0;
        for (std::size_t n = 0; n < samples; ++n) {
            while (snapshotIndex < scheduleCount_ &&
                   controlSchedule_[snapshotIndex].offset == n) {
                const ControlSnapshot& snapshot = controlSchedule_[snapshotIndex];
                syncChannelControlFromSnapshot(channel, snapshot.parameters,
                    snapshot.powerAttackAlpha, snapshot.powerReleaseAlpha,
                    snapshot.gainAttackAlpha, snapshot.gainReleaseAlpha);
                refreshChannelTargets(channel, decisionParallel, decisionContext);
                ++snapshotIndex;
            }
            float x = input != nullptr ? input[n] : 0.0f;
            if (!std::isfinite(x)) x = 0.0f;
            makeBands(state, x, *activeConfig_);
            updateBandPowers(state, channel, *activeConfig_);
            updateCurrentGains(channel);
            const float wet = synthesize(channel, state.band.data());
            const float blend = nextTransitionBlend(channel);
            float y = x + blend * (wet - x);
            if (!std::isfinite(y)) y = x;
            if (accumulate) output[n] += y;
            else output[n] = y;
        }
    }

    double detectorUpdateRateHz() const noexcept {
        return controlStride_ > 0 ? sampleRate_ / static_cast<double>(controlStride_) : 0.0;
    }
    std::size_t detectorStrideSamples() const noexcept { return controlStride_; }
    double powerAttackTimeMs() const noexcept { return powerAttackTimeMs_; }
    double powerReleaseTimeMs() const noexcept { return powerReleaseTimeMs_; }
    double gainAttackTimeMs() const noexcept { return gainAttackTimeMs_; }
    double gainReleaseTimeMs() const noexcept { return gainReleaseTimeMs_; }
    int activeBandCount() const noexcept { return activeConfig_ ? activeConfig_->bandCount : 0; }
#if defined(MUSE_SELF_TEST)
    double maxReconstructionResponse(const std::array<float, kMaxBands>& gains,
                                    int frequencyPoints) const noexcept {
        if (activeConfig_ == nullptr || frequencyPoints < 2) return 0.0;
        double peak = 0.0;
        const int splits = activeConfig_->splitCount;
        const auto& f = activeConfig_->filters;
        for (int bin = 0; bin <= frequencyPoints; ++bin) {
            const double omega = kPi * static_cast<double>(bin) /
                                 static_cast<double>(frequencyPoints);
            const std::complex<double> z1(std::cos(omega), -std::sin(omega));
            const std::complex<double> z2 = z1 * z1;
            std::complex<double> response(gains[static_cast<std::size_t>(activeConfig_->bandCount - 1)], 0.0);
            for (int i = 0; i < splits; ++i) {
                const std::size_t k = static_cast<std::size_t>(i);
                const std::complex<double> numerator =
                    static_cast<double>(f.b0[k]) + static_cast<double>(f.b1[k]) * z1 +
                    static_cast<double>(f.b2[k]) * z2;
                const std::complex<double> denominator =
                    1.0 + static_cast<double>(f.a1[k]) * z1 + static_cast<double>(f.a2[k]) * z2;
                const std::complex<double> lowpass = numerator / denominator;
                const double delta = static_cast<double>(gains[k] - gains[k + 1]);
                response += delta * lowpass;
            }
            peak = std::max(peak, std::abs(response));
        }
        return peak;
    }
#endif
    std::size_t spreadingEntryCount() const noexcept {
        return activeConfig_ != nullptr ? activeConfig_->spreadIndices.size() : 0;
    }
    std::size_t spreadingStorageBytes() const noexcept {
        if (activeConfig_ == nullptr) return 0;
        return activeConfig_->spreadIndices.size() * sizeof(uint16_t) +
               activeConfig_->spreadWeights.size() * sizeof(float) +
               activeConfig_->spreadOffsets.size() * sizeof(uint32_t) +
               activeConfig_->transposeTargets.size() * sizeof(uint16_t) +
               activeConfig_->transposeWeights.size() * sizeof(float) +
               activeConfig_->transposeOffsets.size() * sizeof(uint32_t);
    }
    bool spreadingStorageValid() const noexcept {
        if (activeConfig_ == nullptr) return false;
        const int bands = activeConfig_->bandCount;
        if (activeConfig_->spreadIndices.size() != activeConfig_->spreadWeights.size()) return false;
        if (activeConfig_->transposeTargets.size() != activeConfig_->transposeWeights.size() ||
            activeConfig_->transposeTargets.size() != activeConfig_->spreadIndices.size()) return false;
        if (activeConfig_->spreadOffsets[0] != 0 ||
            activeConfig_->spreadOffsets[static_cast<std::size_t>(bands)] != activeConfig_->spreadIndices.size())
            return false;
        if (activeConfig_->transposeOffsets[0] != 0 ||
            activeConfig_->transposeOffsets[static_cast<std::size_t>(bands)] != activeConfig_->transposeTargets.size())
            return false;
        for (int i = 0; i < bands; ++i) {
            const uint32_t a = activeConfig_->spreadOffsets[static_cast<std::size_t>(i)];
            const uint32_t b = activeConfig_->spreadOffsets[static_cast<std::size_t>(i + 1)];
            if (a > b || b > activeConfig_->spreadIndices.size()) return false;
            const uint32_t ta = activeConfig_->transposeOffsets[static_cast<std::size_t>(i)];
            const uint32_t tb = activeConfig_->transposeOffsets[static_cast<std::size_t>(i + 1)];
            if (ta > tb || tb > activeConfig_->transposeTargets.size()) return false;
        }
        for (int target = 0; target < bands; ++target) {
            const uint32_t first = activeConfig_->spreadOffsets[static_cast<std::size_t>(target)];
            const uint32_t last = activeConfig_->spreadOffsets[static_cast<std::size_t>(target + 1)];
            for (uint32_t j = first; j < last; ++j) {
                const std::size_t masker = activeConfig_->spreadIndices[j];
                if (masker >= static_cast<std::size_t>(bands)) return false;
                const uint32_t tFirst = activeConfig_->transposeOffsets[masker];
                const uint32_t tLast = activeConfig_->transposeOffsets[masker + 1];
                const auto begin = activeConfig_->transposeTargets.begin() + tFirst;
                const auto end = activeConfig_->transposeTargets.begin() + tLast;
                const auto found = std::lower_bound(begin, end, static_cast<uint16_t>(target));
                if (found == end || *found != static_cast<uint16_t>(target)) return false;
                const std::size_t at = static_cast<std::size_t>(found - activeConfig_->transposeTargets.begin());
                if (std::abs(static_cast<double>(activeConfig_->transposeWeights[at] -
                                                 activeConfig_->spreadWeights[j])) > 1.0e-7)
                    return false;
            }
        }
        return true;
    }

    // Parallel decision workers call this on disjoint target/masker ranges.
    void processDecisionRange(int channel, int phase, int partition,
                              int begin, int end) noexcept {
        if (channel < 0 || channel > 1 || activeConfig_ == nullptr) return;
        const int bands = activeConfig_->bandCount;
        begin = std::max(0, std::min(bands, begin));
        end = std::max(begin, std::min(bands, end));
        DecisionScratch& scratch = decisionScratch_[static_cast<std::size_t>(channel)];
        const ChannelControl& control = channelControls_[static_cast<std::size_t>(channel)];
        const Parameters& p = control.current;
        if (phase == 0) {
            double partial = 0.0;
            const double sensitivity = sigmoidUnit(p.sensitivity);
            const double marginDb = 8.0 - 17.0 * sensitivity;
            const double softnessDb = 0.8 + 8.0 * sigmoidUnit(p.softness);
            for (int t = begin; t < end; ++t) {
                const std::size_t ti = static_cast<std::size_t>(t);
                const double priority = control.preferenceCache[ti];
                double totalMaskPower = 0.0;
                if (priority >= 1.0e-5) {
                    const uint32_t first = activeConfig_->spreadOffsets[ti];
                    const uint32_t last = activeConfig_->spreadOffsets[ti + 1];
                    for (uint32_t j = first; j < last; ++j) {
                        const std::size_t m = activeConfig_->spreadIndices[j];
                        totalMaskPower += scratch.powers[m] *
                            static_cast<double>(activeConfig_->spreadWeights[j]);
                    }
                }
                double weight = 0.0;
                if (totalMaskPower > 1.0e-22 && priority >= 1.0e-5) {
                    const double targetDb = powerToDb(scratch.powers[ti]);
                    const double maskDb = powerToDb(totalMaskPower);
                    const double maskedness = logistic((maskDb - targetDb - marginDb) / softnessDb);
                    const double audible = logistic((targetDb + 88.0) / 4.5);
                    weight = priority * audible * maskedness;
                    if (weight < 1.0e-8) weight = 0.0;
                } else {
                    totalMaskPower = 0.0;
                }
                scratch.maskTotals[ti] = totalMaskPower;
                scratch.targetWeights[ti] = weight;
                partial += weight;
            }
            if (partition >= 0 && partition < 2)
                scratch.partialNormalization[static_cast<std::size_t>(partition)] = partial;
            return;
        }
        if (phase == 1) {
            const double normalization = scratch.normalization;
            const double sensitivity = sigmoidUnit(p.sensitivity);
            const double amount = sigmoidUnit(p.amount);
            const double maxCut = clamp(p.maximumReductionDb, 0.0, 9.0);
            const double preference = sigmoidUnit(p.energyPreference);
            const double scoreThreshold = 0.045 + 0.34 * (1.0 - sensitivity);
            const double scoreSoftness = 0.018 + 0.19 * sigmoidUnit(p.softness);
            const double targetActivity = sigmoidUnit(normalization / (normalization + 0.05));
            for (int m = begin; m < end; ++m) {
                const std::size_t mi = static_cast<std::size_t>(m);
                double evidence = 0.0;
                const uint32_t first = activeConfig_->transposeOffsets[mi];
                const uint32_t last = activeConfig_->transposeOffsets[mi + 1];
                if (normalization > 1.0e-10) {
                    for (uint32_t j = first; j < last; ++j) {
                        const std::size_t target = activeConfig_->transposeTargets[j];
                        const double total = scratch.maskTotals[target];
                        if (total <= 1.0e-22) continue;
                        const double contribution = scratch.powers[mi] *
                            static_cast<double>(activeConfig_->transposeWeights[j]);
                        evidence += scratch.targetWeights[target] * contribution / total;
                    }
                }
                scratch.evidence[mi] = evidence;
                const double score = normalization > 1.0e-10
                    ? clamp(evidence / normalization, 0.0, 1.0) : 0.0;
                const double maskGate = logistic((score - scoreThreshold) / scoreSoftness) * targetActivity;
                const double preserve = 1.0 - 0.92 * preference * control.preferenceCache[mi];
                const double cutDb = maxCut * amount * maskGate * preserve;
                scratch.gainDb[mi] = -clamp(cutDb, 0.0, 9.0);
            }
            return;
        }
    }
    const std::array<float, kMaxBands>& gains() const noexcept { return currentGain_[0]; }
    const std::array<float, kMaxBands>& rightGains() const noexcept { return currentGain_[1]; }
    bool prepared() const noexcept { return prepared_; }

private:
    static void resetChannel(ChannelState& ch) noexcept {
        ch.z1.fill(0.0f); ch.z2.fill(0.0f); ch.low.fill(0.0f);
        ch.power.fill(0.0f); ch.band.fill(0.0f);
    }

    static bool buildConfiguration(BandConfiguration& cfg, double sampleRate, int bandCount) noexcept {
        if (bandCount != 32 && bandCount != 64 && bandCount != 128 &&
            bandCount != 256 && bandCount != 512) return false;
        cfg.filters = FilterCoefficients{};
        cfg.centerHz.fill(0.0);
        cfg.centerBark.fill(0.0);
        cfg.spreadOffsets.fill(0);
        cfg.transposeOffsets.fill(0);
        cfg.spreadIndices.clear();
        cfg.spreadWeights.clear();
        cfg.transposeTargets.clear();
        cfg.transposeWeights.clear();
        cfg.bandCount = bandCount;
        cfg.splitCount = bandCount - 1;
        const double fMin = 20.0;
        const double fMax = std::min(20000.0, sampleRate * 0.44);
        if (fMax <= fMin) return false;
        const double rMin = erbRate(fMin);
        const double rMax = erbRate(fMax);
        std::array<double, kMaxSplits> splitHz{};
        for (int i = 0; i < cfg.splitCount; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(cfg.splitCount - 1);
            const double cutoff = inverseErbRate(rMin + t * (rMax - rMin));
            const double w0 = 2.0 * kPi * cutoff / sampleRate;
            const double c = std::cos(w0);
            const double s = std::sin(w0);
            constexpr double q = 0.70710678118654752440;
            const double alpha = s / (2.0 * q);
            const double a0 = 1.0 + alpha;
            cfg.filters.b0[static_cast<std::size_t>(i)] = static_cast<float>(((1.0 - c) * 0.5) / a0);
            cfg.filters.b1[static_cast<std::size_t>(i)] = static_cast<float>((1.0 - c) / a0);
            cfg.filters.b2[static_cast<std::size_t>(i)] = cfg.filters.b0[static_cast<std::size_t>(i)];
            cfg.filters.a1[static_cast<std::size_t>(i)] = static_cast<float>((-2.0 * c) / a0);
            cfg.filters.a2[static_cast<std::size_t>(i)] = static_cast<float>((1.0 - alpha) / a0);
            splitHz[static_cast<std::size_t>(i)] = cutoff;
        }
        cfg.centerHz[0] = splitHz[0] * 0.5;
        for (int i = 1; i < bandCount - 1; ++i)
            cfg.centerHz[static_cast<std::size_t>(i)] = std::sqrt(
                splitHz[static_cast<std::size_t>(i - 1)] * splitHz[static_cast<std::size_t>(i)]);
        cfg.centerHz[static_cast<std::size_t>(bandCount - 1)] = std::sqrt(
            splitHz[static_cast<std::size_t>(cfg.splitCount - 1)] * (sampleRate * 0.5));
        for (int i = 0; i < bandCount; ++i)
            cfg.centerBark[static_cast<std::size_t>(i)] = hzToBark(cfg.centerHz[static_cast<std::size_t>(i)]);

        // Sparse spreading in compact CSR form. The reserve is only a capacity
        // hint; actual storage contains only weights above the -55 dB floor.
        try {
            const std::size_t estimate = static_cast<std::size_t>(bandCount) *
                                         static_cast<std::size_t>(bandCount) / 3u;
            cfg.spreadIndices.reserve(estimate);
            cfg.spreadWeights.reserve(estimate);
            for (int target = 0; target < bandCount; ++target) {
                cfg.spreadOffsets[static_cast<std::size_t>(target)] =
                    static_cast<uint32_t>(cfg.spreadIndices.size());
                for (int masker = 0; masker < bandCount; ++masker) {
                    if (masker == target) continue;
                    const double delta = cfg.centerBark[static_cast<std::size_t>(target)] -
                                         cfg.centerBark[static_cast<std::size_t>(masker)];
                    const double spreadDb = barkSpreadingDb(delta);
                    if (spreadDb < kSpreadFloorDb) continue;
                    cfg.spreadIndices.push_back(static_cast<uint16_t>(masker));
                    cfg.spreadWeights.push_back(static_cast<float>(std::pow(10.0, spreadDb * 0.1)));
                }
                cfg.spreadOffsets[static_cast<std::size_t>(target + 1)] =
                    static_cast<uint32_t>(cfg.spreadIndices.size());
            }

            // Build the transposed view once, outside the audio path. Each
            // masker gets an exclusive row, making evidence accumulation
            // embarrassingly parallel and race-free.
            std::array<uint32_t, kMaxBands> counts{};
            for (int target = 0; target < bandCount; ++target) {
                const uint32_t first = cfg.spreadOffsets[static_cast<std::size_t>(target)];
                const uint32_t last = cfg.spreadOffsets[static_cast<std::size_t>(target + 1)];
                for (uint32_t j = first; j < last; ++j) {
                    const std::size_t masker = cfg.spreadIndices[j];
                    ++counts[masker];
                }
            }
            uint32_t total = 0;
            for (int masker = 0; masker < bandCount; ++masker) {
                cfg.transposeOffsets[static_cast<std::size_t>(masker)] = total;
                total += counts[static_cast<std::size_t>(masker)];
            }
            cfg.transposeOffsets[static_cast<std::size_t>(bandCount)] = total;
            cfg.transposeTargets.resize(total);
            cfg.transposeWeights.resize(total);
            std::array<uint32_t, kMaxBands> cursor{};
            for (int masker = 0; masker < bandCount; ++masker)
                cursor[static_cast<std::size_t>(masker)] =
                    cfg.transposeOffsets[static_cast<std::size_t>(masker)];
            for (int target = 0; target < bandCount; ++target) {
                const uint32_t first = cfg.spreadOffsets[static_cast<std::size_t>(target)];
                const uint32_t last = cfg.spreadOffsets[static_cast<std::size_t>(target + 1)];
                for (uint32_t j = first; j < last; ++j) {
                    const std::size_t masker = cfg.spreadIndices[j];
                    const uint32_t dst = cursor[masker]++;
                    cfg.transposeTargets[dst] = static_cast<uint16_t>(target);
                    cfg.transposeWeights[dst] = cfg.spreadWeights[j];
                }
            }
        } catch (...) {
            cfg.spreadIndices.clear();
            cfg.spreadWeights.clear();
            cfg.transposeTargets.clear();
            cfg.transposeWeights.clear();
            cfg.spreadOffsets.fill(0);
            cfg.transposeOffsets.fill(0);
            return false;
        }
        return true;
    }

    void activateResolution(int mode) noexcept {
        activeMode_ = std::max(0, std::min(4, mode));
        activeConfig_ = &configurations_[static_cast<std::size_t>(activeMode_)];
        resetChannel(left_);
        resetChannel(right_);
        for (auto& gains : targetGain_) gains.fill(1.0f);
        for (auto& gains : currentGain_) gains.fill(1.0f);
        linkedTargetDb_.fill(0.0);
        controlCounter_ = 0;
        transitionSamplesTotal_ = std::max<std::size_t>(64, static_cast<std::size_t>(sampleRate_ * 0.020));
        transitionSamplesRemaining_.fill(transitionSamplesTotal_);
        updateControlRate();
        updateTimeConstants();
        rebuildPreferenceCache();
        syncChannelControlsFromGlobal();
    }

    void updateControlRate() noexcept {
        // Detector cadence is now an explicit user control, not tied to the
        // spectral resolution. The stride remains bounded for stable CPU cost.
        const double refreshHz = clamp(current_.detectorRateHz, 250.0, 1500.0);
        const double maximumStride = std::max(32.0, std::ceil(sampleRate_ / 250.0));
        controlStride_ = static_cast<std::size_t>(clamp(
            std::round(sampleRate_ / refreshHz), 32.0, maximumStride));
        parameterAlpha_ = 1.0 - std::exp(-static_cast<double>(controlStride_) /
                                         (0.015 * sampleRate_));
    }

    void smoothParameters() noexcept {
        const auto sm = [this](double& c, double t) { c += parameterAlpha_ * (t - c); };
        sm(current_.amount, target_.amount);
        sm(current_.sensitivity, target_.sensitivity);
        sm(current_.softness, target_.softness);
        sm(current_.maximumReductionDb, target_.maximumReductionDb);
        sm(current_.energyPreference, target_.energyPreference);
        sm(current_.body, target_.body);
        sm(current_.vocal, target_.vocal);
        sm(current_.presence, target_.presence);
        sm(current_.articulation, target_.articulation);
        sm(current_.air, target_.air);
        sm(current_.focusWidthBark, target_.focusWidthBark);
        sm(current_.responsiveness, target_.responsiveness);
        sm(current_.stereoLink, target_.stereoLink);
        sm(current_.detectorRateHz, target_.detectorRateHz);
        current_.resolutionMode = target_.resolutionMode;
    }

    void updateTimeConstants() noexcept {
        const double response = sigmoidUnit(current_.responsiveness);
        const double attackMs = 7.0 - 5.5 * response;
        const double releaseMs = 150.0 - 115.0 * response;
        powerAttackTimeMs_ = attackMs;
        powerReleaseTimeMs_ = releaseMs;
        powerAttackAlpha_ = static_cast<float>(timeAlpha(sampleRate_, powerAttackTimeMs_));
        powerReleaseAlpha_ = static_cast<float>(timeAlpha(sampleRate_, powerReleaseTimeMs_));
        // Power envelopes run every sample and retain pure millisecond timing.
        // Gain targets are refreshed only at the decision cadence. Prevent the
        // gain attack from settling faster than one decision interval, which
        // would leave it following a stale target until the next refresh.
        const double decisionIntervalMs = 1000.0 * static_cast<double>(controlStride_) / sampleRate_;
        gainAttackTimeMs_ = std::max(std::max(1.5, attackMs * 0.8), decisionIntervalMs);
        gainReleaseTimeMs_ = std::max(std::max(12.0, releaseMs * 0.8), decisionIntervalMs);
        gainAttackAlpha_ = static_cast<float>(timeAlpha(sampleRate_, gainAttackTimeMs_));
        gainReleaseAlpha_ = static_cast<float>(timeAlpha(sampleRate_, gainReleaseTimeMs_));
    }

    void refreshControlParameters() noexcept {
        smoothParameters();
        updateControlRate();
        updateTimeConstants();
    }

    void syncChannelControlFromSnapshot(int channel, const Parameters& parameters,
                                        float powerAttack, float powerRelease,
                                        float gainAttack, float gainRelease) noexcept {
        if (channel < 0 || channel > 1) return;
        ChannelControl& control = channelControls_[static_cast<std::size_t>(channel)];
        const Parameters& old = control.current;
        const bool preferenceChanged = !control.preferenceInitialized ||
            old.body != parameters.body || old.vocal != parameters.vocal ||
            old.presence != parameters.presence || old.articulation != parameters.articulation ||
            old.air != parameters.air || old.focusWidthBark != parameters.focusWidthBark;
        control.current = parameters;
        control.powerAttackAlpha = powerAttack;
        control.powerReleaseAlpha = powerRelease;
        control.gainAttackAlpha = gainAttack;
        control.gainReleaseAlpha = gainRelease;
        if (preferenceChanged) rebuildChannelPreferenceCache(channel);
    }

    void syncChannelControlsFromGlobal() noexcept {
        for (int channel = 0; channel < 2; ++channel) {
            ChannelControl& control = channelControls_[static_cast<std::size_t>(channel)];
            control.current = current_;
            control.preferenceCache = preferenceCache_;
            control.powerAttackAlpha = powerAttackAlpha_;
            control.powerReleaseAlpha = powerReleaseAlpha_;
            control.gainAttackAlpha = gainAttackAlpha_;
            control.gainReleaseAlpha = gainReleaseAlpha_;
            control.preferenceInitialized = true;
        }
    }

    void rebuildChannelPreferenceCache(int channel) noexcept {
        if (channel < 0 || channel > 1 || activeConfig_ == nullptr) return;
        ChannelControl& control = channelControls_[static_cast<std::size_t>(channel)];
        const Parameters& p = control.current;
        const double width = std::max(0.35, p.focusWidthBark);
        const double strengths[5] = {
            sigmoidUnit(p.body), sigmoidUnit(p.vocal), sigmoidUnit(p.presence),
            sigmoidUnit(p.articulation), sigmoidUnit(p.air)
        };
        constexpr double centers[5] = {4.8, 8.6, 12.5, 16.3, 21.0};
        for (int i = 0; i < activeConfig_->bandCount; ++i) {
            const double z = activeConfig_->centerBark[static_cast<std::size_t>(i)];
            double uncovered = 1.0;
            for (int pIndex = 0; pIndex < 5; ++pIndex) {
                const double d = (z - centers[pIndex]) / width;
                const double weight = strengths[pIndex] * std::exp(-0.5 * d * d);
                uncovered *= (1.0 - weight);
            }
            control.preferenceCache[static_cast<std::size_t>(i)] = clamp(1.0 - uncovered, 0.0, 1.0);
        }
        for (int i = activeConfig_->bandCount; i < static_cast<int>(kMaxBands); ++i)
            control.preferenceCache[static_cast<std::size_t>(i)] = 0.0;
        control.preferenceInitialized = true;
    }

    void buildControlSchedule(std::size_t samples) noexcept {
        scheduleCount_ = 0;
        std::size_t offset = 0;
        while (offset < samples && scheduleCount_ < kMaxControlSnapshots) {
            if (controlCounter_ == 0) {
                refreshControlParameters();
                ControlSnapshot& snapshot = controlSchedule_[scheduleCount_++];
                snapshot.offset = offset;
                snapshot.parameters = current_;
                snapshot.powerAttackAlpha = powerAttackAlpha_;
                snapshot.powerReleaseAlpha = powerReleaseAlpha_;
                snapshot.gainAttackAlpha = gainAttackAlpha_;
                snapshot.gainReleaseAlpha = gainReleaseAlpha_;
            }
            const std::size_t toControl = controlStride_ > controlCounter_
                ? controlStride_ - controlCounter_ : 1;
            const std::size_t count = std::min(samples - offset, toControl);
            controlCounter_ += count;
            if (controlCounter_ >= controlStride_) controlCounter_ = 0;
            offset += count;
        }
    }

    void makeBands(ChannelState& ch, float input, const BandConfiguration& cfg) noexcept {
        const auto& f = cfg.filters;
        if (avxEnabled_) {
            muse_avx_make_bands(cfg.splitCount, cfg.bandCount, input,
                f.b0.data(), f.b1.data(), f.b2.data(), f.a1.data(), f.a2.data(),
                ch.z1.data(), ch.z2.data(), ch.low.data(), ch.band.data());
            return;
        }
        // Independent filters are stored in SoA form; /O2 can vectorize this loop.
#if defined(_MSC_VER)
#pragma loop(ivdep)
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
        for (int i = 0; i < cfg.splitCount; ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            const float y = f.b0[k] * input + ch.z1[k];
            const float nextZ1 = f.b1[k] * input - f.a1[k] * y + ch.z2[k];
            const float nextZ2 = f.b2[k] * input - f.a2[k] * y;
            ch.z1[k] = nextZ1;
            ch.z2[k] = nextZ2;
            ch.low[k] = y;
        }
        ch.band[0] = ch.low[0];
#if defined(_MSC_VER)
#pragma loop(ivdep)
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
        for (int i = 1; i < cfg.splitCount; ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            ch.band[k] = ch.low[k] - ch.low[k - 1];
        }
        ch.band[static_cast<std::size_t>(cfg.bandCount - 1)] = input - ch.low[static_cast<std::size_t>(cfg.splitCount - 1)];
    }

    void updateBandPowers(ChannelState& ch, int channel, const BandConfiguration& cfg) noexcept {
        const ChannelControl& control = channelControls_[static_cast<std::size_t>(channel)];
        if (avxEnabled_) {
            muse_avx_update_powers(cfg.bandCount, ch.band.data(), ch.power.data(),
                                       control.powerAttackAlpha, control.powerReleaseAlpha);
            return;
        }
#if defined(_MSC_VER)
#pragma loop(ivdep)
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
        for (int i = 0; i < cfg.bandCount; ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            const float p = ch.band[k] * ch.band[k];
            const float alpha = p > ch.power[k] ? control.powerAttackAlpha : control.powerReleaseAlpha;
            ch.power[k] += alpha * (p - ch.power[k]);
            if (!std::isfinite(ch.power[k]) || ch.power[k] < 0.0f) ch.power[k] = 0.0f;
        }
    }

    void rebuildPreferenceCache() noexcept {
        if (activeConfig_ == nullptr) return;
        const double width = std::max(0.35, current_.focusWidthBark);
        const double strengths[5] = {
            sigmoidUnit(current_.body), sigmoidUnit(current_.vocal),
            sigmoidUnit(current_.presence), sigmoidUnit(current_.articulation),
            sigmoidUnit(current_.air)
        };
        constexpr double centers[5] = {4.8, 8.6, 12.5, 16.3, 21.0};
        for (int i = 0; i < activeConfig_->bandCount; ++i) {
            const double z = activeConfig_->centerBark[static_cast<std::size_t>(i)];
            double uncovered = 1.0;
            for (int p = 0; p < 5; ++p) {
                const double d = (z - centers[p]) / width;
                const double weight = strengths[p] * std::exp(-0.5 * d * d);
                uncovered *= (1.0 - weight);
            }
            preferenceCache_[static_cast<std::size_t>(i)] = clamp(1.0 - uncovered, 0.0, 1.0);
        }
        for (int i = activeConfig_->bandCount; i < static_cast<int>(kMaxBands); ++i)
            preferenceCache_[static_cast<std::size_t>(i)] = 0.0;
    }

    double preferenceWeight(std::size_t band) const noexcept {
        return preferenceCache_[band];
    }

    void computeSubtractiveTargets(const std::array<double, kMaxBands>& powers,
                                   std::array<double, kMaxBands>& gainDb, int channel,
                                   DecisionParallel decisionParallel = nullptr,
                                   void* decisionContext = nullptr) noexcept {
        if (channel < 0 || channel > 1 || activeConfig_ == nullptr) return;
        const int bands = activeConfig_->bandCount;
        DecisionScratch& scratch = decisionScratch_[static_cast<std::size_t>(channel)];
        scratch.powers = powers;
        scratch.targetWeights.fill(0.0);
        scratch.maskTotals.fill(0.0);
        scratch.evidence.fill(0.0);
        scratch.gainDb.fill(0.0);
        scratch.partialNormalization.fill(0.0);
        scratch.normalization = 0.0;

        const bool useParallel = decisionParallel != nullptr && bands >= 256;
        if (useParallel) decisionParallel(decisionContext, this, channel, 0, bands);
        else processDecisionRange(channel, 0, 0, 0, bands);

        scratch.normalization = scratch.partialNormalization[0] + scratch.partialNormalization[1];
        if (useParallel) decisionParallel(decisionContext, this, channel, 1, bands);
        else processDecisionRange(channel, 1, 0, 0, bands);

        gainDb = scratch.gainDb;
    }

    void refreshControlState() noexcept {
        refreshControlParameters();
        rebuildPreferenceCache();
        syncChannelControlsFromGlobal();
        linkedTargetDb_.fill(0.0);
        if (current_.stereoLink <= 1.0e-6) return;

        std::array<double, kMaxBands> linkedPower{};
        for (int i = 0; i < activeConfig_->bandCount; ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            linkedPower[k] = 0.5 * (static_cast<double>(left_.power[k]) +
                                    static_cast<double>(right_.power[k]));
        }
        computeSubtractiveTargets(linkedPower, linkedTargetDb_, 0);
    }

    void refreshChannelTargets(int channel, DecisionParallel decisionParallel,
                               void* decisionContext) noexcept {
        ChannelState& state = channel == 0 ? left_ : right_;
        std::array<double, kMaxBands> channelPower{};
        std::array<double, kMaxBands> localGainDb{};
        for (int i = 0; i < activeConfig_->bandCount; ++i)
            channelPower[static_cast<std::size_t>(i)] = state.power[static_cast<std::size_t>(i)];
        computeSubtractiveTargets(channelPower, localGainDb, channel,
                                  decisionParallel, decisionContext);

        const double link = clamp(channelControls_[static_cast<std::size_t>(channel)].current.stereoLink, 0.0, 1.0);
        for (int i = 0; i < activeConfig_->bandCount; ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            // Interpolate the requested attenuation in dB. At 0% each channel
            // follows only its own detector; at 100% both use the same curve.
            const double gainDb = localGainDb[k] * (1.0 - link) + linkedTargetDb_[k] * link;
            targetGain_[static_cast<std::size_t>(channel)][k] = clampf(
                static_cast<float>(dbToLinear(gainDb)), static_cast<float>(dbToLinear(-9.0)), 1.0f);
        }
        for (int i = activeConfig_->bandCount; i < static_cast<int>(kMaxBands); ++i)
            targetGain_[static_cast<std::size_t>(channel)][static_cast<std::size_t>(i)] = 1.0f;
    }

    void updateCurrentGains(int channel) noexcept {
        const int bands = activeConfig_->bandCount;
        auto& current = currentGain_[static_cast<std::size_t>(channel)];
        const auto& target = targetGain_[static_cast<std::size_t>(channel)];
#if defined(_MSC_VER)
#pragma loop(ivdep)
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
        for (int i = 0; i < bands; ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            const ChannelControl& control = channelControls_[static_cast<std::size_t>(channel)];
            const float alpha = target[k] < current[k]
                ? control.gainAttackAlpha : control.gainReleaseAlpha;
            current[k] += alpha * (target[k] - current[k]);
            current[k] = clampf(current[k], 0.0f, 1.0f);
        }
    }

    float processChannelSample(int channel, float input) noexcept {
        ChannelState& state = channel == 0 ? left_ : right_;
        if (!std::isfinite(input)) input = 0.0f;
        makeBands(state, input, *activeConfig_);
        updateBandPowers(state, channel, *activeConfig_);
        updateCurrentGains(channel);
        const float wet = synthesize(channel, state.band.data());
        const float blend = nextTransitionBlend(channel);
        float output = input + blend * (wet - input);
        if (!std::isfinite(output)) output = input;
        return output;
    }

    float nextTransitionBlend(int channel) noexcept {
        auto& remaining = transitionSamplesRemaining_[static_cast<std::size_t>(channel)];
        if (remaining == 0 || transitionSamplesTotal_ == 0) return 1.0f;
        const float blend = 1.0f - static_cast<float>(remaining) /
                                  static_cast<float>(transitionSamplesTotal_);
        --remaining;
        return static_cast<float>(sigmoidUnit(blend));
    }

    float synthesize(int channel, const float* bandsData) const noexcept {
        float y = 0.0f;
        const int bands = activeConfig_->bandCount;
        const auto& gains = currentGain_[static_cast<std::size_t>(channel)];
        for (int i = 0; i < bands; ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            y += bandsData[k] * gains[k];
        }
        return std::isfinite(y) ? y : 0.0f;
    }

    double sampleRate_ = 44100.0;
    std::size_t controlStride_ = 64;
    std::size_t controlCounter_ = 0;
    std::size_t transitionSamplesTotal_ = 0;
    std::array<std::size_t, 2> transitionSamplesRemaining_{};
    double parameterAlpha_ = 0.0;
    float powerAttackAlpha_ = 0.0f;
    float powerReleaseAlpha_ = 0.0f;
    float gainAttackAlpha_ = 0.0f;
    float gainReleaseAlpha_ = 0.0f;
    double powerAttackTimeMs_ = 7.0;
    double powerReleaseTimeMs_ = 150.0;
    double gainAttackTimeMs_ = 1.5;
    double gainReleaseTimeMs_ = 28.0;
    bool prepared_ = false;
    bool parametersInitialized_ = false;
    bool avxEnabled_ = false;
    int activeMode_ = 1;
    Parameters target_{};
    Parameters current_{};
    std::array<BandConfiguration, kResolutionCount> configurations_{};
    const BandConfiguration* activeConfig_ = nullptr;
    ChannelState left_{};
    ChannelState right_{};
    std::array<std::array<float, kMaxBands>, 2> targetGain_{};
    std::array<std::array<float, kMaxBands>, 2> currentGain_{};
    std::array<ChannelControl, 2> channelControls_{};
    std::array<DecisionScratch, 2> decisionScratch_{};
    std::array<ControlSnapshot, kMaxControlSnapshots> controlSchedule_{};
    std::size_t scheduleCount_ = 0;
    std::array<double, kMaxBands> preferenceCache_{};
    std::array<double, kMaxBands> linkedTargetDb_{};
};

} // namespace muse_perceptual

#ifdef MUSE_SELF_TEST

#include <iostream>
#include <memory>
#include <random>
#include <thread>

namespace {
int failures = 0;
std::unique_ptr<muse_perceptual::StereoEngine> makeTestEngine(bool enableAvx = true) noexcept {
    return std::unique_ptr<muse_perceptual::StereoEngine>(
        new (std::nothrow) muse_perceptual::StereoEngine(enableAvx));
}

struct ParallelTestJob {
    muse_perceptual::StereoEngine* engine = nullptr;
    int channel = 0;
    const float* input = nullptr;
    float* output = nullptr;
    std::size_t samples = 0;
    bool accumulate = false;
    bool refreshTargets = false;
    bool scheduledIndependent = false;
    void* decisionContext = nullptr;
};
struct ParallelTestContext {
    std::array<ParallelTestJob, 2> jobs{};
    int waitCount = 0;
    int submitCount = 0;
    bool parallelDecision = false;
};

void runParallelTestDecisionPhase(void* opaque, muse_perceptual::StereoEngine* engine,
                                  int channel, int phase, int bandCount) noexcept {
    (void)opaque;
    if (engine == nullptr) return;
    const int midpoint = bandCount / 2;
    std::thread first([=]() noexcept {
        engine->processDecisionRange(channel, phase, 0, 0, midpoint);
    });
    std::thread second([=]() noexcept {
        engine->processDecisionRange(channel, phase, 1, midpoint, bandCount);
    });
    first.join();
    second.join();
}

void queueParallelTestChannel(void* opaque, muse_perceptual::StereoEngine* engine,
                              int channel, const float* input, float* output,
                              std::size_t samples, bool accumulate,
                              bool refreshTargets, bool scheduledIndependent) noexcept {
    auto* context = static_cast<ParallelTestContext*>(opaque);
    if (context == nullptr || channel < 0 || channel > 1) return;
    context->jobs[static_cast<std::size_t>(channel)] = {
        engine, channel, input, output, samples, accumulate, refreshTargets, scheduledIndependent, context
    };
    ++context->submitCount;
}

void runParallelTestChannel(ParallelTestJob job) noexcept {
    if (job.engine != nullptr)
        job.engine->processChannelBlock(job.channel, job.input, job.output,
                                        job.samples, job.accumulate, job.refreshTargets,
                                        static_cast<ParallelTestContext*>(job.decisionContext)->parallelDecision
                                            ? &runParallelTestDecisionPhase : nullptr,
                                        job.decisionContext, job.scheduledIndependent);
}

void waitParallelTestChannels(void* opaque) noexcept {
    auto* context = static_cast<ParallelTestContext*>(opaque);
    if (context == nullptr) return;
    ++context->waitCount;
    std::thread left(runParallelTestChannel, context->jobs[0]);
    std::thread right(runParallelTestChannel, context->jobs[1]);
    left.join();
    right.join();
}

void check(bool condition, const char* name) {
    std::cout << (condition ? "[PASS] " : "[FAIL] ") << name << '\n';
    if (!condition) ++failures;
}

muse_perceptual::Parameters strongParams(int mode = 2) {
    muse_perceptual::Parameters p;
    p.amount = 1.0; p.sensitivity = 1.0; p.softness = 0.10;
    p.maximumReductionDb = 9.0; p.energyPreference = 0.0;
    p.body = 0.0; p.vocal = 1.0; p.presence = 0.0;
    p.articulation = 0.0; p.air = 0.0; p.focusWidthBark = 0.6;
    p.responsiveness = 1.0; p.resolutionMode = mode;
    return p;
}
}

int main() {
    using namespace muse_perceptual;
    std::cout << "AVX1 runtime dispatch: " << (cpuSupportsAvx() ? "available" : "baseline fallback") << '\n';
#if MUSE_HAS_MXCSR
    {
        const unsigned int originalMxcsr = _mm_getcsr();
        const unsigned int baselineMxcsr = originalMxcsr & ~(kMxcsrDazMask | kMxcsrFtzMask);
        _mm_setcsr(baselineMxcsr);
        bool denormalFlagsEnabled = false;
        bool dazFlushesSubnormalInputs = false;
        bool ftzFlushesSubnormalResults = false;
        {
            ScopedDenormalMode denormalMode;
            const unsigned int activeMxcsr = _mm_getcsr();
            denormalFlagsEnabled = (activeMxcsr & (kMxcsrDazMask | kMxcsrFtzMask)) ==
                                   (kMxcsrDazMask | kMxcsrFtzMask);

            // Amplify the smallest subnormal to a normal-sized result: if DAZ
            // is active, the subnormal input is treated as zero before multiply.
            const uint32_t smallestSubnormalBits = 1u;
            float smallestSubnormal = 0.0f;
            std::memcpy(&smallestSubnormal, &smallestSubnormalBits, sizeof(smallestSubnormal));
            volatile float subnormalInput = smallestSubnormal;
            volatile float amplifiedInput = subnormalInput * 1.0e30f;
            dazFlushesSubnormalInputs = amplifiedInput == 0.0f;

            // Halving the smallest normal float produces a subnormal result;
            // FTZ should write zero instead of retaining that tiny value.
            volatile float smallestNormal = std::numeric_limits<float>::min();
            volatile float underflowResult = smallestNormal * 0.5f;
            ftzFlushesSubnormalResults = underflowResult == 0.0f;
        }
        const unsigned int restoredMxcsr = _mm_getcsr();
        _mm_setcsr(originalMxcsr);
        check(denormalFlagsEnabled, "DSP denormal guard enables both DAZ and FTZ");
        check(dazFlushesSubnormalInputs, "DAZ treats subnormal input operands as zero");
        check(ftzFlushesSubnormalResults, "FTZ flushes subnormal floating-point results to zero");
        check(restoredMxcsr == baselineMxcsr, "scoped denormal guard restores the caller MXCSR exactly");
    }
    {
        unsigned int workerOriginalMxcsr = 0;
        unsigned int workerActiveMxcsr = 0;
        unsigned int workerRestoredMxcsr = 0;
        std::thread worker([&]() noexcept {
            workerOriginalMxcsr = _mm_getcsr();
            {
                ScopedDenormalMode denormalMode;
                workerActiveMxcsr = _mm_getcsr();
            }
            workerRestoredMxcsr = _mm_getcsr();
        });
        worker.join();
        check((workerActiveMxcsr & (kMxcsrDazMask | kMxcsrFtzMask)) ==
              (kMxcsrDazMask | kMxcsrFtzMask),
              "denormal mode is independently enabled on a worker thread");
        check(workerRestoredMxcsr == workerOriginalMxcsr,
              "worker thread restores its own MXCSR after the DSP scope");
    }
#endif
    check(resolutionBandCount(0) == 32 && resolutionBandCount(1) == 64 &&
          resolutionBandCount(2) == 128 && resolutionBandCount(3) == 256 &&
          resolutionBandCount(4) == 512,
          "32/64/128/256/512 selectable resolutions");
    check(resolutionModeFromBandCount(32.0) == 0 && resolutionModeFromBandCount(64.0) == 1 &&
          resolutionModeFromBandCount(128.0) == 2 && resolutionModeFromBandCount(256.0) == 3 &&
          resolutionModeFromBandCount(512.0) == 4,
          "numeric resolution entry snaps to all five supported band counts");
    check(Parameters{}.stereoLink == 0.0, "Stereo Link factory default is independent (0%)");
    check(std::abs(Parameters{}.detectorRateHz - 1000.0) < 1.0e-9,
          "detector refresh factory default is 1000 Hz");

    for (int mode = 0; mode < static_cast<int>(kResolutionCount); ++mode) {
        auto rateEngine = makeTestEngine();
        if (!rateEngine) { std::cerr << "[FATAL] Could not allocate detector-rate test engine.\n"; return 2; }
        check(rateEngine->prepare(48000.0), "prepare detector-rate test engine");
        Parameters rateParams;
        rateParams.resolutionMode = mode;
        rateParams.detectorRateHz = 1200.0;
        rateEngine->setParameters(rateParams);
        float rateL = 0.0f, rateR = 0.0f;
        rateEngine->processStereo(0.05f, 0.04f, rateL, rateR);
        check(rateEngine->activeBandCount() == resolutionBandCount(mode),
              "selected analysis resolution is active");
        check(rateEngine->spreadingStorageValid() &&
              rateEngine->spreadingEntryCount() < static_cast<std::size_t>(resolutionBandCount(mode) * resolutionBandCount(mode)),
              "CSR spreading offsets and compact entries are valid");
        if (mode == 4)
            check(rateEngine->spreadingStorageBytes() < 1300000,
                  "512-band forward + transposed CSR tables remain compact");
        check(std::abs(rateEngine->detectorUpdateRateHz() - 1200.0) < 0.5,
              "detector update rate is independent of 32/64/128/256/512 resolution");
    }

    auto fastDecision = makeTestEngine();
    auto slowDecision = makeTestEngine();
    if (!fastDecision || !slowDecision) {
        std::cerr << "[FATAL] Could not allocate tick-aware attack test engines.\n"; return 2;
    }
    Parameters fastDecisionParams; fastDecisionParams.responsiveness = 1.0;
    fastDecisionParams.detectorRateHz = 1500.0;
    Parameters slowDecisionParams = fastDecisionParams; slowDecisionParams.detectorRateHz = 250.0;
    fastDecision->setParameters(fastDecisionParams); slowDecision->setParameters(slowDecisionParams);
    check(fastDecision->prepare(48000.0) && slowDecision->prepare(48000.0),
          "prepare tick-aware attack test engines");
    check(std::abs(fastDecision->gainAttackTimeMs() - 1.5) < 0.01 &&
          std::abs(slowDecision->gainAttackTimeMs() - 4.0) < 0.01,
          "gain attack respects the decision interval");
    check(std::abs(fastDecision->powerAttackTimeMs() - slowDecision->powerAttackTimeMs()) < 1.0e-9 &&
          std::abs(fastDecision->powerReleaseTimeMs() - slowDecision->powerReleaseTimeMs()) < 1.0e-9,
          "power-envelope attack and release remain independent of decision cadence");
    check(std::abs(slowDecision->gainReleaseTimeMs() - fastDecision->gainReleaseTimeMs()) < 1.0e-9 &&
          slowDecision->gainReleaseTimeMs() >= 28.0,
          "gain release stays time-based above the decision-interval floor");

    auto maxRateEngine = makeTestEngine();
    if (!maxRateEngine) { std::cerr << "[FATAL] Could not allocate max detector-rate test engine.\n"; return 2; }
    maxRateEngine->prepare(48000.0);
    Parameters maxRateParams; maxRateParams.detectorRateHz = 1500.0;
    maxRateEngine->setParameters(maxRateParams);
    check(maxRateEngine->detectorStrideSamples() == 32 &&
          std::abs(maxRateEngine->detectorUpdateRateHz() - 1500.0) < 0.5,
          "1500 Hz detector rate respects the 32-sample minimum stride at 48 kHz");

    auto lowRateEngine = makeTestEngine();
    if (!lowRateEngine) { std::cerr << "[FATAL] Could not allocate minimum detector-rate test engine.\n"; return 2; }
    lowRateEngine->prepare(192000.0);
    Parameters lowRateParams; lowRateParams.detectorRateHz = 250.0;
    lowRateEngine->setParameters(lowRateParams);
    check(std::abs(lowRateEngine->detectorUpdateRateHz() - 250.0) < 0.5,
          "250 Hz detector rate remains available at 192 kHz sample rate");

    auto identity = makeTestEngine();
    if (!identity) { std::cerr << "[FATAL] Could not allocate identity test engine.\n"; return 2; }
    check(identity->prepare(48000.0), "prepare all resolutions at 48 kHz");
    Parameters neutral;
    neutral.amount = 0.0;
    neutral.body = neutral.vocal = neutral.presence = neutral.articulation = neutral.air = 0.0;
    neutral.resolutionMode = 1;
    identity->setParameters(neutral);
    double identityError = 0.0;
    for (int n = 0; n < 24000; ++n) {
        const float xL = 0.2f * std::sin(static_cast<float>(2.0 * kPi * 440.0 * n / 48000.0)) +
                         0.1f * std::sin(static_cast<float>(2.0 * kPi * 6300.0 * n / 48000.0));
        const float xR = 0.18f * std::sin(static_cast<float>(2.0 * kPi * 330.0 * n / 48000.0));
        float yL = 0.0f, yR = 0.0f;
        identity->processStereo(xL, xR, yL, yR);
        identityError = std::max(identityError, std::abs(static_cast<double>(yL - xL)));
        identityError = std::max(identityError, std::abs(static_cast<double>(yR - xR)));
    }
    check(identityError < 2.0e-6, "unity reconstruction is transparent (float DSP)");

    auto identity512 = makeTestEngine(false); // exercise the baseline path at maximum resolution
    if (!identity512) { std::cerr << "[FATAL] Could not allocate 512-band unity test engine.\n"; return 2; }
    check(identity512->prepare(48000.0), "prepare 512-band unity reconstruction test");
    neutral.resolutionMode = 4;
    identity512->setParameters(neutral);
    double identity512Error = 0.0;
    for (int n = 0; n < 12000; ++n) {
        const float t = static_cast<float>(n) / 48000.0f;
        const float xL = 0.2f * std::sin(static_cast<float>(2.0 * kPi * 440.0 * t)) +
                         0.1f * std::sin(static_cast<float>(2.0 * kPi * 6300.0 * t));
        const float xR = 0.18f * std::sin(static_cast<float>(2.0 * kPi * 330.0 * t));
        float yL = 0.0f, yR = 0.0f;
        identity512->processStereo(xL, xR, yL, yR);
        identity512Error = std::max(identity512Error, std::abs(static_cast<double>(yL - xL)));
        identity512Error = std::max(identity512Error, std::abs(static_cast<double>(yR - xR)));
    }
    check(identity512Error < 8.0e-6, "512-band scalar fallback unity reconstruction remains transparent");

    for (int mode = 0; mode < static_cast<int>(kResolutionCount); ++mode) {
        auto engine = makeTestEngine();
        if (!engine) { std::cerr << "[FATAL] Could not allocate resolution test engine.\n"; return 2; }
        check(engine->prepare(48000.0), "prepare selectable-band engine");
        Parameters p = strongParams(mode);
        engine->setParameters(p);
        double peak = 0.0;
        const int renderSamples = mode >= 4 ? 12000 : (mode >= 3 ? 24000 : 48000 * 2);
        for (int n = 0; n < renderSamples; ++n) {
            const float t = static_cast<float>(n) / 48000.0f;
            const float x = 0.18f * std::sin(static_cast<float>(2.0 * kPi * 1000.0 * t)) +
                            0.50f * std::sin(static_cast<float>(2.0 * kPi * 1450.0 * t)) +
                            0.11f * std::sin(static_cast<float>(2.0 * kPi * 3600.0 * t));
            float yL = 0.0f, yR = 0.0f;
            engine->processStereo(x, x, yL, yR);
            peak = std::max(peak, std::abs(static_cast<double>(yL)));
        }
        bool bounded = true, anyCut = false;
        for (float g : engine->gains()) { bounded = bounded && g >= 0.0f && g <= 1.0f; anyCut = anyCut || g < 0.999f; }
        check(engine->activeBandCount() == resolutionBandCount(mode), "active resolution switches cleanly");
        check(bounded, "all per-band gains remain subtractive");
        check(anyCut, "masking model produces attenuation");
        const double responsePeak = engine->maxReconstructionResponse(engine->gains(), 4096);
        check(std::isfinite(responsePeak) && responsePeak <= 1.02,
              "measured steady-state reconstruction remains within +0.17 dB for the regression signal");
        check(std::isfinite(peak) && peak < 2.0, "finite output at selected resolution");
    }

    {
        // Resolution should not make the detector progressively blind to the
        // same broadband material simply because each analysis cell is narrower.
        std::mt19937 noiseRng(0x512BADDu);
        std::uniform_real_distribution<float> noiseDist(-0.35f, 0.35f);
        std::array<float, 24000> noise{};
        for (auto& sample : noise) sample = noiseDist(noiseRng);
        std::array<double, kResolutionCount> noiseRmsRatios{};
        bool allNoiseModesValid = true;
        for (int mode = 0; mode < static_cast<int>(kResolutionCount); ++mode) {
            auto noiseEngine = makeTestEngine();
            if (!noiseEngine || !noiseEngine->prepare(48000.0)) { allNoiseModesValid = false; break; }
            noiseEngine->setParameters(strongParams(mode));
            double inputEnergy = 0.0, outputEnergy = 0.0;
            for (std::size_t n = 0; n < noise.size(); ++n) {
                float left = 0.0f, right = 0.0f;
                noiseEngine->processStereo(noise[n], noise[n], left, right);
                if (n >= 6000) {
                    inputEnergy += static_cast<double>(noise[n]) * noise[n];
                    outputEnergy += static_cast<double>(left) * left;
                }
            }
            noiseRmsRatios[static_cast<std::size_t>(mode)] = std::sqrt(outputEnergy / inputEnergy);
        }
        const auto limits = std::minmax_element(noiseRmsRatios.begin(), noiseRmsRatios.end());
        check(allNoiseModesValid && (*limits.second - *limits.first) < 0.02,
              "broadband noise attenuation remains calibrated across 32-512-band resolutions");
    }

    {
        auto responseProbe = makeTestEngine();
        if (!responseProbe || !responseProbe->prepare(48000.0)) {
            std::cerr << "[FATAL] Could not allocate reconstruction-response test engine.\n";
            return 2;
        }
        Parameters responseParams; responseParams.resolutionMode = 4; responseParams.amount = 0.0;
        responseProbe->setParameters(responseParams);
        std::array<float, kMaxBands> profile{};
        bool responseProfilesBounded = true;
        for (int pattern = 0; pattern < 8; ++pattern) {
            for (int i = 0; i < 512; ++i) {
                const double x = static_cast<double>(i) / 511.0;
                float gain = 1.0f;
                if (pattern == 0) gain = (i & 1) ? 0.25f : 1.0f;
                if (pattern == 1) gain = static_cast<float>(1.0 - 0.8 * std::exp(-0.5 * std::pow((x - 0.5) / 0.06, 2.0)));
                if (pattern == 2) gain = static_cast<float>(0.55 + 0.45 * (0.5 + 0.5 * std::sin(20.0 * x)));
                if (pattern == 3) gain = (i >= 100 && i < 200) ? 0.2f : 1.0f;
                if (pattern == 4) gain = (i >= 250 && i < 400) ? 0.2f : 1.0f;
                if (pattern == 5) gain = (i >= 250 && i < 260) ? 0.0f : 1.0f;
                if (pattern == 6) gain = static_cast<float>(0.8 + 0.2 * std::sin(5.0 * x));
                if (pattern == 7) gain = static_cast<float>(0.3 + 0.7 * std::pow(x, 4.0));
                profile[static_cast<std::size_t>(i)] = gain;
            }
            const double responsePeak = responseProbe->maxReconstructionResponse(profile, 4096);
            responseProfilesBounded = responseProfilesBounded && std::isfinite(responsePeak) && responsePeak <= 1.02;
        }
        check(responseProfilesBounded,
              "512-band response scan covers alternating, notched, smooth and stepped gain profiles");
    }

    auto switching = makeTestEngine();
    if (!switching) { std::cerr << "[FATAL] Could not allocate switching test engine.\n"; return 2; }
    switching->prepare(48000.0);
    Parameters p = strongParams(0);
    switching->setParameters(p);
    float l = 0.0f, r = 0.0f;
    for (int n = 0; n < 2048; ++n) switching->processStereo(0.1f, 0.1f, l, r);
    p.resolutionMode = 4;
    switching->setParameters(p);
    bool switchFinite = true;
    for (int n = 0; n < 4096; ++n) {
        switching->processStereo(0.1f, 0.1f, l, r);
        switchFinite = switchFinite && std::isfinite(l) && std::isfinite(r);
    }
    check(switchFinite, "resolution switch remains finite");
    check(switching->activeBandCount() == 512, "runtime resolution switch reaches 512 bands");

    auto stereo = makeTestEngine();
    if (!stereo) { std::cerr << "[FATAL] Could not allocate stereo test engine.\n"; return 2; }
    stereo->prepare(44100.0);
    stereo->setParameters(strongParams(1));
    bool identical = true;
    for (int n = 0; n < 12000; ++n) {
        const float x = 0.2f * std::sin(static_cast<float>(2.0 * kPi * 987.0 * n / 44100.0));
        float a = 0.0f, b = 0.0f;
        stereo->processStereo(x, x, a, b);
        identical = identical && std::abs(a - b) < 1.0e-6f;
    }
    check(identical, "independent processing preserves matching identical channels");

    auto independent = makeTestEngine();
    if (!independent) { std::cerr << "[FATAL] Could not allocate independent-channel test engine.\n"; return 2; }
    independent->prepare(48000.0);
    Parameters independentParams = strongParams(1);
    independentParams.stereoLink = 0.0;
    independent->setParameters(independentParams);
    for (int n = 0; n < 48000 * 2; ++n) {
        const float t = static_cast<float>(n) / 48000.0f;
        const float left = 0.18f * std::sin(static_cast<float>(2.0 * kPi * 1000.0 * t)) +
                           0.50f * std::sin(static_cast<float>(2.0 * kPi * 1450.0 * t)) +
                           0.11f * std::sin(static_cast<float>(2.0 * kPi * 3600.0 * t));
        const float right = 0.18f * std::sin(static_cast<float>(2.0 * kPi * 1000.0 * t));
        float outL = 0.0f, outR = 0.0f;
        independent->processStereo(left, right, outL, outR);
    }
    double independentDifference = 0.0;
    for (std::size_t i = 0; i < kMaxBands; ++i)
        independentDifference = std::max(independentDifference,
            std::abs(static_cast<double>(independent->gains()[i] - independent->rightGains()[i])));
    check(independentDifference > 1.0e-4,
          "0% Stereo Link keeps L/R masking gains independent");

    auto fullyLinked = makeTestEngine();
    if (!fullyLinked) { std::cerr << "[FATAL] Could not allocate linked-channel test engine.\n"; return 2; }
    fullyLinked->prepare(48000.0);
    Parameters linkedParams = independentParams;
    linkedParams.stereoLink = 1.0;
    fullyLinked->setParameters(linkedParams);
    for (int n = 0; n < 48000 * 2; ++n) {
        const float t = static_cast<float>(n) / 48000.0f;
        const float left = 0.18f * std::sin(static_cast<float>(2.0 * kPi * 1000.0 * t)) +
                           0.50f * std::sin(static_cast<float>(2.0 * kPi * 1450.0 * t)) +
                           0.11f * std::sin(static_cast<float>(2.0 * kPi * 3600.0 * t));
        const float right = 0.18f * std::sin(static_cast<float>(2.0 * kPi * 1000.0 * t));
        float outL = 0.0f, outR = 0.0f;
        fullyLinked->processStereo(left, right, outL, outR);
    }
    double linkedDifference = 0.0;
    for (std::size_t i = 0; i < kMaxBands; ++i)
        linkedDifference = std::max(linkedDifference,
            std::abs(static_cast<double>(fullyLinked->gains()[i] - fullyLinked->rightGains()[i])));
    check(linkedDifference < 1.0e-6,
          "100% Stereo Link applies one shared gain curve to both channels");

    auto partiallyLinked = makeTestEngine();
    if (!partiallyLinked) { std::cerr << "[FATAL] Could not allocate partial-link test engine.\n"; return 2; }
    partiallyLinked->prepare(48000.0);
    Parameters partialParams = independentParams;
    partialParams.stereoLink = 0.5;
    partiallyLinked->setParameters(partialParams);
    for (int n = 0; n < 48000 * 2; ++n) {
        const float t = static_cast<float>(n) / 48000.0f;
        const float left = 0.18f * std::sin(static_cast<float>(2.0 * kPi * 1000.0 * t)) +
                           0.50f * std::sin(static_cast<float>(2.0 * kPi * 1450.0 * t)) +
                           0.11f * std::sin(static_cast<float>(2.0 * kPi * 3600.0 * t));
        const float right = 0.18f * std::sin(static_cast<float>(2.0 * kPi * 1000.0 * t));
        float outL = 0.0f, outR = 0.0f;
        partiallyLinked->processStereo(left, right, outL, outR);
    }
    double partialDifference = 0.0;
    for (std::size_t i = 0; i < kMaxBands; ++i)
        partialDifference = std::max(partialDifference,
            std::abs(static_cast<double>(partiallyLinked->gains()[i] - partiallyLinked->rightGains()[i])));
    check(partialDifference > linkedDifference + 1.0e-4 &&
          partialDifference < independentDifference,
          "50% Stereo Link smoothly interpolates between independent and full link");

    auto highRate = makeTestEngine();
    if (!highRate) { std::cerr << "[FATAL] Could not allocate high-rate test engine.\n"; return 2; }
    check(highRate->prepare(192000.0), "prepare 192 kHz with all configurations");
    highRate->setParameters(strongParams(4));
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> noise(-0.7f, 0.7f);
    bool finite = true;
    for (int n = 0; n < 48000; ++n) {
        float a = 0.0f, b = 0.0f;
        highRate->processStereo(noise(rng), noise(rng), a, b);
        finite = finite && std::isfinite(a) && std::isfinite(b);
    }
    check(finite, "finite processing at 192 kHz");

    auto sampleReference = makeTestEngine(false); // force baseline path for comparison
    auto blockEngine = makeTestEngine(true);       // AVX1 path when available
    if (!sampleReference || !blockEngine) {
        std::cerr << "[FATAL] Could not allocate block-path test engines.\n";
        return 2;
    }
    check(sampleReference->prepare(48000.0) && blockEngine->prepare(48000.0),
          "prepare sample and block-path comparison engines");
    const Parameters blockParams = strongParams(1);
    sampleReference->setParameters(blockParams);
    blockEngine->setParameters(blockParams);
    std::array<float, 64> inL{}, inR{}, outL{}, outR{};
    double blockPathError = 0.0;
    for (int base = 0; base < 64 * 48; base += 64) {
        for (int n = 0; n < 64; ++n) {
            const float t = static_cast<float>(base + n) / 48000.0f;
            inL[static_cast<std::size_t>(n)] = 0.22f * std::sin(static_cast<float>(2.0 * kPi * 930.0 * t)) +
                                                0.08f * std::sin(static_cast<float>(2.0 * kPi * 5100.0 * t));
            inR[static_cast<std::size_t>(n)] = 0.19f * std::sin(static_cast<float>(2.0 * kPi * 640.0 * t)) +
                                                0.05f * std::sin(static_cast<float>(2.0 * kPi * 3700.0 * t));
        }
        blockEngine->processStereoBlock(inL.data(), inR.data(), outL.data(), outR.data(), 64, false);
        for (int n = 0; n < 64; ++n) {
            float refL = 0.0f, refR = 0.0f;
            sampleReference->processStereo(inL[static_cast<std::size_t>(n)],
                                           inR[static_cast<std::size_t>(n)], refL, refR);
            blockPathError = std::max(blockPathError, std::abs(static_cast<double>(refL - outL[static_cast<std::size_t>(n)])));
            blockPathError = std::max(blockPathError, std::abs(static_cast<double>(refR - outR[static_cast<std::size_t>(n)])));
        }
    }
    check(blockPathError < 2.0e-6, "block processing matches sample processing");

    auto parallelReference = makeTestEngine(false);
    auto parallelEngine = makeTestEngine(true);
    if (!parallelReference || !parallelEngine) {
        std::cerr << "[FATAL] Could not allocate parallel-dispatch test engines.\n";
        return 2;
    }
    check(parallelReference->prepare(48000.0) && parallelEngine->prepare(48000.0),
          "prepare parallel-dispatch comparison engines");
    parallelReference->setParameters(blockParams);
    parallelEngine->setParameters(blockParams);
    ParallelTestContext parallelContext{};
    double parallelPathError = 0.0;
    for (int base = 0; base < 64 * 48; base += 64) {
        for (int n = 0; n < 64; ++n) {
            const float t = static_cast<float>(base + n) / 48000.0f;
            inL[static_cast<std::size_t>(n)] = 0.22f * std::sin(static_cast<float>(2.0 * kPi * 930.0 * t)) +
                                                0.08f * std::sin(static_cast<float>(2.0 * kPi * 5100.0 * t));
            inR[static_cast<std::size_t>(n)] = 0.19f * std::sin(static_cast<float>(2.0 * kPi * 640.0 * t)) +
                                                0.05f * std::sin(static_cast<float>(2.0 * kPi * 3700.0 * t));
        }
        parallelEngine->processStereoBlock(inL.data(), inR.data(), outL.data(), outR.data(), 64,
            false, &queueParallelTestChannel, &waitParallelTestChannels, &parallelContext);
        for (int n = 0; n < 64; ++n) {
            float refL = 0.0f, refR = 0.0f;
            parallelReference->processStereo(inL[static_cast<std::size_t>(n)],
                                             inR[static_cast<std::size_t>(n)], refL, refR);
            parallelPathError = std::max(parallelPathError,
                std::abs(static_cast<double>(refL - outL[static_cast<std::size_t>(n)])));
            parallelPathError = std::max(parallelPathError,
                std::abs(static_cast<double>(refR - outR[static_cast<std::size_t>(n)])));
        }
    }
    check(parallelPathError < 2.0e-6,
          "two concurrent channel workers match the synchronous reference");

    auto irregularSample = makeTestEngine(false);
    auto irregularBlock = makeTestEngine(true);
    if (!irregularSample || !irregularBlock) {
        std::cerr << "[FATAL] Could not allocate irregular-block test engines.\n";
        return 2;
    }
    check(irregularSample->prepare(48000.0) && irregularBlock->prepare(48000.0),
          "prepare irregular host-block scheduling engines");
    Parameters irregularParams = strongParams(4);
    irregularParams.detectorRateHz = 1375.0;
    irregularSample->setParameters(irregularParams);
    irregularBlock->setParameters(irregularParams);
    const std::array<std::size_t, 9> irregularSizes = {1, 17, 31, 64, 97, 257, 511, 1024, 43};
    double irregularBlockError = 0.0;
    int sampleNumber = 0;
    for (const std::size_t blockSamples : irregularSizes) {
        std::vector<float> srcL(blockSamples), srcR(blockSamples), dstL(blockSamples), dstR(blockSamples);
        for (std::size_t n = 0; n < blockSamples; ++n, ++sampleNumber) {
            const float t = static_cast<float>(sampleNumber) / 48000.0f;
            srcL[n] = 0.20f * std::sin(static_cast<float>(2.0 * kPi * 1040.0 * t)) +
                      0.21f * std::sin(static_cast<float>(2.0 * kPi * 1510.0 * t));
            srcR[n] = 0.17f * std::sin(static_cast<float>(2.0 * kPi * 620.0 * t)) +
                      0.07f * std::sin(static_cast<float>(2.0 * kPi * 4100.0 * t));
        }
        irregularBlock->processStereoBlock(srcL.data(), srcR.data(), dstL.data(), dstR.data(),
                                           blockSamples, false);
        for (std::size_t n = 0; n < blockSamples; ++n) {
            float refL = 0.0f, refR = 0.0f;
            irregularSample->processStereo(srcL[n], srcR[n], refL, refR);
            irregularBlockError = std::max(irregularBlockError,
                std::abs(static_cast<double>(refL - dstL[n])));
            irregularBlockError = std::max(irregularBlockError,
                std::abs(static_cast<double>(refR - dstR[n])));
        }
    }
    check(irregularBlockError < 3.0e-6,
          "scheduled whole-block path preserves tick timing across irregular host block sizes");

    // Exercise the independent-mode whole-block dispatch and the masker-centric
    // transpose path with real concurrent test workers. The host-side coordinator
    // must submit exactly two jobs and wait once for the complete block.
    auto decisionReference = makeTestEngine(false);
    auto decisionParallel = makeTestEngine(true);
    if (!decisionReference || !decisionParallel) {
        std::cerr << "[FATAL] Could not allocate parallel-decision test engines.\n";
        return 2;
    }
    check(decisionReference->prepare(48000.0) && decisionParallel->prepare(48000.0),
          "prepare 512-band parallel-decision comparison engines");
    Parameters decisionParams = strongParams(4);
    decisionParams.detectorRateHz = 250.0;
    decisionReference->setParameters(decisionParams);
    decisionParallel->setParameters(decisionParams);
    // One complete detector interval is enough to exercise both parallel CSR
    // phases while keeping sanitizer runs independent of thread-startup cost.
    std::vector<float> longL(192), longR(192), refLongL(192), refLongR(192),
                       parLongL(192), parLongR(192);
    for (std::size_t n = 0; n < longL.size(); ++n) {
        const float t = static_cast<float>(n) / 48000.0f;
        longL[n] = 0.20f * std::sin(static_cast<float>(2.0 * kPi * 980.0 * t)) +
                   0.34f * std::sin(static_cast<float>(2.0 * kPi * 1480.0 * t)) +
                   0.08f * std::sin(static_cast<float>(2.0 * kPi * 5200.0 * t));
        longR[n] = 0.18f * std::sin(static_cast<float>(2.0 * kPi * 700.0 * t)) +
                   0.10f * std::sin(static_cast<float>(2.0 * kPi * 3300.0 * t));
    }
    decisionReference->processStereoBlock(longL.data(), longR.data(),
        refLongL.data(), refLongR.data(), longL.size(), false);
    ParallelTestContext decisionContext{};
    decisionContext.parallelDecision = true;
    decisionParallel->processStereoBlock(longL.data(), longR.data(),
        parLongL.data(), parLongR.data(), longL.size(), false,
        &queueParallelTestChannel, &waitParallelTestChannels, &decisionContext);
    double decisionParallelError = 0.0;
    for (std::size_t n = 0; n < longL.size(); ++n) {
        decisionParallelError = std::max(decisionParallelError,
            std::abs(static_cast<double>(refLongL[n] - parLongL[n])));
        decisionParallelError = std::max(decisionParallelError,
            std::abs(static_cast<double>(refLongR[n] - parLongR[n])));
    }
    check(decisionParallelError < 3.0e-6,
          "two-way parallel CSR decision matches serial 512-band reference");
    check(decisionContext.submitCount == 2 && decisionContext.waitCount == 1,
          "independent mode submits whole channel blocks and waits once per block");

    check(std::abs(sigmoidUnit(0.451) - sigmoidUnit(0.45)) < 0.01,
          "sigmoid control mapping remains continuous");
    std::cout << "Failures: " << failures << '\n';
    return failures == 0 ? 0 : 1;
}

#else // VST2 Windows build

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "vst.h"
#include <windows.h>
#include <commctrl.h>
#include <process.h>

#if !defined(_WIN32)
#error This VST2 target currently builds Windows x64 only.
#endif

namespace {
using namespace muse_perceptual;
HINSTANCE gModuleInstance = nullptr;

constexpr int32_t kVstEffectOpen = 0;
constexpr int32_t kVstEffectClose = 1;
constexpr int32_t kVstEffectSetProgram = 2;
constexpr int32_t kVstEffectGetProgram = 3;
constexpr int32_t kVstEffectSetProgramName = 4;
constexpr int32_t kVstEffectGetProgramName = 5;
constexpr int32_t kVstEffectGetParamLabel = 6;
constexpr int32_t kVstEffectGetParamDisplay = 7;
constexpr int32_t kVstEffectGetParamName = 8;
constexpr int32_t kVstEffectSetSampleRate = 10;
constexpr int32_t kVstEffectSetBlockSize = 11;
constexpr int32_t kVstEffectMainsChanged = 12;
constexpr int32_t kVstEffectEditGetRect = 13;
constexpr int32_t kVstEffectEditOpen = 14;
constexpr int32_t kVstEffectEditClose = 15;
constexpr int32_t kVstEffectEditIdle = 19;
constexpr int32_t kVstEffectGetCategory = 35;
constexpr int32_t kVstEffectSetBypass = 44;
constexpr int32_t kVstEffectGetEffectName = 45;
constexpr int32_t kVstEffectGetVendorString = 47;
constexpr int32_t kVstEffectGetProductString = 48;
constexpr int32_t kVstEffectGetVendorVersion = 49;
constexpr int32_t kVstEffectCanDo = 51;
constexpr int32_t kVstEffectGetVstVersion = 58;
constexpr int32_t kVstHostGetVstVersion = 1;
constexpr int32_t kVstVersion24 = 2400;
constexpr int32_t kPluginVersion = 1303;
// v1.3.3 adds a 16th parameter (Worker Priority) for unified thread priority control.
// The original 15-parameter layout and normalized ranges are unchanged from v1.2;
// retain MESA so existing host states remain compatible.
constexpr int32_t kPluginUniqueId = static_cast<int32_t>(VST_FOURCC('M', 'E', 'S', 'A'));
constexpr std::size_t kParameterCount = 16;
constexpr UINT kEditorRefreshMessage = WM_APP + 0x62;
constexpr int kEditorSliderIdBase = 5200;
constexpr int kEditorPresetButtonId = 5400;
constexpr int kParametersPerColumn = 8; // 16 parameters / 2 columns = 8 per column
constexpr int kEditorWidth = 1120;
constexpr int kEditorHeight = 480; // Increased to accommodate 16 parameters (2 rows of 8)
constexpr int kEditorFirstRowY = 112;
constexpr int kEditorRowHeight = 39;
constexpr wchar_t kEditorClassName[] = L"muse_Perceptual_v133_Editor";

// Worker priority levels: 0=Idle, 1=Below Normal, 2=Normal, 3=Above Normal, 4=Highest
constexpr int kWorkerPriorityLevels = 5;
constexpr std::array<const wchar_t*, kWorkerPriorityLevels> kWorkerPriorityNames = {
    L"Idle", L"Below Normal", L"Normal", L"Above Normal", L"Highest"
};
constexpr std::array<int, kWorkerPriorityLevels> kWorkerPriorityValues = {
    THREAD_PRIORITY_IDLE, THREAD_PRIORITY_BELOW_NORMAL, THREAD_PRIORITY_NORMAL,
    THREAD_PRIORITY_ABOVE_NORMAL, THREAD_PRIORITY_HIGHEST
};

inline int normalizedToWorkerPriority(float v) noexcept {
    v = clampf(v, 0.0f, 1.0f);
    if (v < 0.1f) return 0;
    if (v < 0.3f) return 1;
    if (v < 0.5f) return 2;
    if (v < 0.7f) return 3;
    return 4;
}
inline float workerPriorityToNormalized(int level) noexcept {
    level = std::max(0, std::min(4, level));
    return static_cast<float>(level) * 0.25f;
}

constexpr std::array<const wchar_t*, kParameterCount> kParameterNamesWide = {
    L"Focus Amount", L"Mask Sensitivity", L"Sigmoid Softness", L"Maximum Reduction",
    L"Energy Preference", L"Body", L"Vocal", L"Presence", L"Articulation", L"Air",
    L"Focus Width", L"Responsiveness", L"Band Resolution", L"Stereo Link", L"Detector Update",
    L"Worker Priority"
};
constexpr std::array<const char*, kParameterCount> kParameterNames = {
    "Focus Amount", "Mask Sensitivity", "Sigmoid Softness", "Maximum Reduction",
    "Energy Preference", "Body", "Vocal", "Presence", "Articulation", "Air",
    "Focus Width", "Responsiveness", "Band Resolution", "Stereo Link", "Detector Update",
    "Worker Priority"
};
constexpr std::array<const char*, kParameterCount> kParameterLabels = {
    "%", "%", "%", "dB", "%", "%", "%", "%", "%", "%", "Bark", "%", "bands", "%", "Hz", "level"
};

inline float clamp01f(float x) noexcept {
    if (!std::isfinite(x)) return 0.0f;
    return std::max(0.0f, std::min(1.0f, x));
}
inline double denormalize(double n, double lo, double hi) noexcept {
    return lo + static_cast<double>(clamp01f(static_cast<float>(n))) * (hi - lo);
}
inline float normalizeLinear(double v, double lo, double hi) noexcept {
    return static_cast<float>(clamp((v - lo) / (hi - lo), 0.0, 1.0));
}

void copyVstString(void* destination, std::size_t capacity, const char* source) noexcept {
    if (destination == nullptr || capacity == 0) return;
    auto* out = static_cast<char*>(destination);
    std::memset(out, 0, capacity);
    if (source == nullptr) return;
    const std::size_t count = std::min(capacity - 1, std::strlen(source));
    std::memcpy(out, source, count);
}

class MuseEsPerceptualPlugin final {
public:
    explicit MuseEsPerceptualPlugin(vst_host_callback_t host) noexcept : host_(host) {
        effect_.magic_number = VST_MAGICNUMBER;
        effect_.control = &dispatch;
        effect_.process = &processAccumulating;
        effect_.set_parameter = &setParameter;
        effect_.get_parameter = &getParameter;
        effect_.num_programs = 1;
        effect_.num_params = static_cast<int32_t>(kParameterCount);
        effect_.num_inputs = 2;
        effect_.num_outputs = 2;
        effect_.flags = VST_EFFECT_FLAG_EDITOR | VST_EFFECT_FLAG_SUPPORTS_FLOAT;
        effect_._unknown_00 = nullptr;
        effect_._unknown_01 = nullptr;
        effect_.delay = 0;
        effect_._unknown_02 = 0;
        effect_._unknown_03 = 0;
        effect_.input_output_ratio = 1.0f;
        effect_.effect_internal = this;
        effect_.host_internal = nullptr;
        effect_.unique_id = kPluginUniqueId;
        effect_.version = kPluginVersion;
        effect_.process_float = &processReplacing;
        effect_.process_double = nullptr;
        std::memset(effect_._unknown, 0, sizeof(effect_._unknown));
        setDefaults();
        const bool initialPrepared = engine_.prepare(44100.0);
        preparedRate_.store(initialPrepared ? 44100.0 : 0.0, std::memory_order_release);
        prepared_.store(initialPrepared, std::memory_order_release);
        startParallelWorkers();
    }
    ~MuseEsPerceptualPlugin() noexcept {
        stopParallelWorkers();
        closeEditor();
    }
    vst_effect_t* effect() noexcept { return &effect_; }

private:
    struct DecisionWorker {
        HANDLE workEvent = nullptr;
        HANDLE doneEvent = nullptr;
        HANDLE thread = nullptr;
        std::atomic<bool> stopping{false};
        StereoEngine* engine = nullptr;
        int channel = 0;
        int phase = 0;
        int partition = 0;
        int begin = 0;
        int end = 0;
    };

    struct ChannelWorker {
        int channel = 0;
        HANDLE workEvent = nullptr;
        HANDLE doneEvent = nullptr;
        HANDLE thread = nullptr;
        std::atomic<bool> stopping{false};
        std::atomic<uint64_t> jobGeneration{0};
        std::atomic<uint64_t> completedGeneration{0};
        StereoEngine* engine = nullptr;
        const float* input = nullptr;
        float* output = nullptr;
        std::size_t samples = 0;
        bool accumulate = false;
        bool refreshTargets = false;
        bool scheduledIndependent = false;
        std::array<DecisionWorker, 2> decisionWorkers{};
        bool decisionWorkersReady = false;
    };

    static MuseEsPerceptualPlugin* fromEffect(vst_effect_t* e) noexcept {
        return e == nullptr ? nullptr : static_cast<MuseEsPerceptualPlugin*>(e->effect_internal);
    }
    static unsigned __stdcall channelWorkerProc(void* context) noexcept {
        auto* worker = static_cast<ChannelWorker*>(context);
        if (worker == nullptr) return 0;
        // The channel worker owns the long-running filter-bank/envelope DSP.
        // Set FTZ/DAZ once for this worker thread, including its entire lifetime.
        ScopedDenormalMode denormalMode;
        for (;;) {
            const DWORD wr = WaitForSingleObject(worker->workEvent, INFINITE);
            if (wr != WAIT_OBJECT_0) break;
            if (worker->stopping.load(std::memory_order_acquire)) break;
            const uint64_t generation = worker->jobGeneration.load(std::memory_order_acquire);
            StereoEngine* engine = worker->engine;
            if (engine != nullptr) {
                engine->processChannelBlock(worker->channel, worker->input, worker->output,
                    worker->samples, worker->accumulate, worker->refreshTargets,
                    worker->decisionWorkersReady ? &runParallelDecisionPhase : nullptr,
                    worker, worker->scheduledIndependent);
            }
            worker->completedGeneration.store(generation, std::memory_order_release);
            SetEvent(worker->doneEvent);
        }
        return 0;
    }

    static unsigned __stdcall decisionWorkerProc(void* context) noexcept {
        auto* worker = static_cast<DecisionWorker*>(context);
        if (worker == nullptr) return 0;
        // Decision math also runs on a separate persistent thread. MXCSR is
        // thread-local, so give these workers the same denormal protection.
        ScopedDenormalMode denormalMode;
        for (;;) {
            const DWORD wr = WaitForSingleObject(worker->workEvent, INFINITE);
            if (wr != WAIT_OBJECT_0) break;
            if (worker->stopping.load(std::memory_order_acquire)) break;
            StereoEngine* engine = worker->engine;
            if (engine != nullptr)
                engine->processDecisionRange(worker->channel, worker->phase,
                    worker->partition, worker->begin, worker->end);
            SetEvent(worker->doneEvent);
        }
        return 0;
    }

    static void runParallelDecisionPhase(void* context, StereoEngine* engine,
                                         int channel, int phase, int bandCount) noexcept {
        auto* parent = static_cast<ChannelWorker*>(context);
        if (parent == nullptr || engine == nullptr || !parent->decisionWorkersReady) {
            if (engine != nullptr) engine->processDecisionRange(channel, phase, 0, 0, bandCount);
            return;
        }
        const int midpoint = bandCount / 2;
        const int starts[2] = {0, midpoint};
        const int ends[2] = {midpoint, bandCount};
        HANDLE activeDoneEvents[2]{};
        DWORD activeCount = 0;
        for (int i = 0; i < 2; ++i) {
            DecisionWorker& worker = parent->decisionWorkers[static_cast<std::size_t>(i)];
            worker.engine = engine;
            worker.channel = channel;
            worker.phase = phase;
            worker.partition = i;
            worker.begin = starts[i];
            worker.end = ends[i];
            if (!SetEvent(worker.workEvent)) {
                engine->processDecisionRange(channel, phase, i, starts[i], ends[i]);
            } else {
                activeDoneEvents[activeCount++] = worker.doneEvent;
            }
        }
        if (activeCount > 0)
            WaitForMultipleObjects(activeCount, activeDoneEvents, TRUE, INFINITE);
    }

    bool startDecisionWorkers(ChannelWorker& parent) noexcept {
        bool success = true;
        for (int i = 0; i < 2; ++i) {
            DecisionWorker& worker = parent.decisionWorkers[static_cast<std::size_t>(i)];
            worker.channel = parent.channel;
            worker.partition = i;
            worker.workEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            worker.doneEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (worker.workEvent == nullptr || worker.doneEvent == nullptr) {
                success = false;
                break;
            }
            const uintptr_t thread = _beginthreadex(nullptr, 0, &decisionWorkerProc, &worker, 0, nullptr);
            worker.thread = reinterpret_cast<HANDLE>(thread);
            if (worker.thread == nullptr) {
                success = false;
                break;
            }
        }
        if (!success) {
            for (auto& worker : parent.decisionWorkers) {
                if (worker.thread != nullptr) {
                    worker.stopping.store(true, std::memory_order_release);
                    if (worker.workEvent != nullptr) SetEvent(worker.workEvent);
                }
            }
            for (auto& worker : parent.decisionWorkers) {
                if (worker.thread != nullptr) {
                    WaitForSingleObject(worker.thread, INFINITE);
                    CloseHandle(worker.thread);
                    worker.thread = nullptr;
                }
                if (worker.workEvent != nullptr) CloseHandle(worker.workEvent);
                if (worker.doneEvent != nullptr) CloseHandle(worker.doneEvent);
                worker.workEvent = worker.doneEvent = nullptr;
                worker.stopping.store(false, std::memory_order_relaxed);
            }
            return false;
        }
        parent.decisionWorkersReady = true;
        return true;
    }

    void startParallelWorkers() noexcept {
        if (GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) < 2) return;
        bool success = true;
        for (int i = 0; i < 2; ++i) {
            ChannelWorker& worker = workers_[static_cast<std::size_t>(i)];
            worker.channel = i;
            worker.workEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            worker.doneEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (worker.workEvent == nullptr || worker.doneEvent == nullptr) {
                success = false;
                break;
            }
            const uintptr_t thread = _beginthreadex(nullptr, 0, &channelWorkerProc, &worker, 0, nullptr);
            worker.thread = reinterpret_cast<HANDLE>(thread);
            if (worker.thread == nullptr) {
                success = false;
                break;
            }
        }
        if (!success) {
            stopParallelWorkers();
            return;
        }
        for (auto& worker : workers_)
            SetThreadPriority(worker.thread, THREAD_PRIORITY_ABOVE_NORMAL);
        parallelWorkerReady_ = true;

        // Four additional persistent workers are useful on 4+ logical CPUs.
        // Each audio-channel worker owns a pair, avoiding cross-channel job
        // queue contention and keeping all decision scratch channel-local.
        if (GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) >= 4) {
            bool decisionReady = true;
            for (auto& worker : workers_)
                decisionReady = startDecisionWorkers(worker) && decisionReady;
            if (!decisionReady) {
                for (auto& worker : workers_) stopDecisionWorkers(worker);
            }
        }
    }

    static void stopDecisionWorkers(ChannelWorker& parent) noexcept {
        parent.decisionWorkersReady = false;
        for (auto& worker : parent.decisionWorkers) {
            if (worker.thread != nullptr) {
                worker.stopping.store(true, std::memory_order_release);
                if (worker.workEvent != nullptr) SetEvent(worker.workEvent);
            }
        }
        for (auto& worker : parent.decisionWorkers) {
            if (worker.thread != nullptr) {
                WaitForSingleObject(worker.thread, INFINITE);
                CloseHandle(worker.thread);
                worker.thread = nullptr;
            }
            if (worker.workEvent != nullptr) CloseHandle(worker.workEvent);
            if (worker.doneEvent != nullptr) CloseHandle(worker.doneEvent);
            worker.workEvent = worker.doneEvent = nullptr;
            worker.stopping.store(false, std::memory_order_relaxed);
        }
    }

    void stopParallelWorkers() noexcept {
        parallelWorkerReady_ = false;
        // Drain/stop channel jobs before tearing down their decision helpers.
        // A channel worker may currently be waiting for a decision phase;
        // keeping those helpers alive until both audio workers exit avoids
        // a shutdown-time wait on an already-terminated analyzer.
        for (auto& worker : workers_) {
            if (worker.thread != nullptr) {
                worker.stopping.store(true, std::memory_order_release);
                if (worker.workEvent != nullptr) SetEvent(worker.workEvent);
            }
        }
        for (auto& worker : workers_) {
            if (worker.thread != nullptr) {
                WaitForSingleObject(worker.thread, INFINITE);
                CloseHandle(worker.thread);
                worker.thread = nullptr;
            }
            if (worker.workEvent != nullptr) CloseHandle(worker.workEvent);
            if (worker.doneEvent != nullptr) CloseHandle(worker.doneEvent);
            worker.workEvent = worker.doneEvent = nullptr;
        }
        for (auto& worker : workers_) stopDecisionWorkers(worker);
    }

    static void submitChannel(void* context, StereoEngine* engine, int channel,
                              const float* input, float* output, std::size_t samples,
                              bool accumulate, bool refreshTargets,
                              bool scheduledIndependent) noexcept {
        auto* self = static_cast<MuseEsPerceptualPlugin*>(context);
        if (self == nullptr || !self->parallelWorkerReady_ || channel < 0 || channel > 1) return;
        ChannelWorker& worker = self->workers_[static_cast<std::size_t>(channel)];
        worker.engine = engine;
        worker.input = input;
        worker.output = output;
        worker.samples = samples;
        worker.accumulate = accumulate;
        worker.refreshTargets = refreshTargets;
        worker.scheduledIndependent = scheduledIndependent;
        worker.jobGeneration.fetch_add(1, std::memory_order_release);
        if (!SetEvent(worker.workEvent)) {
            // Fail closed: complete this channel synchronously and release the
            // same completion gate so the other channel's worker can finish.
            engine->processChannelBlock(channel, input, output, samples, accumulate,
                refreshTargets, worker.decisionWorkersReady ? &runParallelDecisionPhase : nullptr,
                &worker, scheduledIndependent);
            worker.completedGeneration.store(
                worker.jobGeneration.load(std::memory_order_relaxed), std::memory_order_release);
            SetEvent(worker.doneEvent);
        }
    }

    static void waitChannels(void* context) noexcept {
        auto* self = static_cast<MuseEsPerceptualPlugin*>(context);
        if (self == nullptr || !self->parallelWorkerReady_) return;
        HANDLE completed[2] = { self->workers_[0].doneEvent, self->workers_[1].doneEvent };
        WaitForMultipleObjects(2, completed, TRUE, INFINITE);
        (void)self->workers_[0].completedGeneration.load(std::memory_order_acquire);
        (void)self->workers_[1].completedGeneration.load(std::memory_order_acquire);
    }

    void setDefaults() noexcept {
        parameters_[0].store(normalizeLinear(42.0, 0.0, 100.0));
        parameters_[1].store(normalizeLinear(58.0, 0.0, 100.0));
        parameters_[2].store(normalizeLinear(62.0, 0.0, 100.0));
        parameters_[3].store(normalizeLinear(3.5, 0.0, 9.0));
        parameters_[4].store(normalizeLinear(72.0, 0.0, 100.0));
        parameters_[5].store(normalizeLinear(22.0, 0.0, 100.0));
        parameters_[6].store(normalizeLinear(62.0, 0.0, 100.0));
        parameters_[7].store(normalizeLinear(42.0, 0.0, 100.0));
        parameters_[8].store(normalizeLinear(25.0, 0.0, 100.0));
        parameters_[9].store(normalizeLinear(8.0, 0.0, 100.0));
        parameters_[10].store(normalizeLinear(1.35, 0.35, 4.0));
        parameters_[11].store(normalizeLinear(72.0, 0.0, 100.0));
        parameters_[12].store(0.25f); // 64 bands default (five discrete resolutions)
        parameters_[13].store(0.0f); // independent L/R by default
        parameters_[14].store(normalizeLinear(1000.0, 250.0, 1500.0)); // 1 kHz default detector cadence
        parameters_[15].store(workerPriorityToNormalized(3)); // Above Normal (matches v1.3.2 behavior)
    }
    Parameters makeParameters() const noexcept {
        Parameters p;
        p.amount = parameters_[0].load(std::memory_order_acquire);
        p.sensitivity = parameters_[1].load(std::memory_order_acquire);
        p.softness = parameters_[2].load(std::memory_order_acquire);
        p.maximumReductionDb = denormalize(parameters_[3].load(std::memory_order_acquire), 0.0, 9.0);
        p.energyPreference = parameters_[4].load(std::memory_order_acquire);
        p.body = parameters_[5].load(std::memory_order_acquire);
        p.vocal = parameters_[6].load(std::memory_order_acquire);
        p.presence = parameters_[7].load(std::memory_order_acquire);
        p.articulation = parameters_[8].load(std::memory_order_acquire);
        p.air = parameters_[9].load(std::memory_order_acquire);
        p.focusWidthBark = denormalize(parameters_[10].load(std::memory_order_acquire), 0.35, 4.0);
        p.responsiveness = parameters_[11].load(std::memory_order_acquire);
        p.resolutionMode = normalizedToResolution(parameters_[12].load(std::memory_order_acquire));
        p.stereoLink = parameters_[13].load(std::memory_order_acquire);
        p.detectorRateHz = denormalize(parameters_[14].load(std::memory_order_acquire), 250.0, 1500.0);
        return p;
    }
    static int editorRowY(std::size_t i) noexcept {
        return kEditorFirstRowY + static_cast<int>(i % kParametersPerColumn) * kEditorRowHeight;
    }
    static bool registerEditorClass() noexcept {
        INITCOMMONCONTROLSEX icc{}; icc.dwSize = sizeof(icc); icc.dwICC = ICC_BAR_CLASSES;
        if (!InitCommonControlsEx(&icc)) return false;
        WNDCLASSEXW wc{}; wc.cbSize = sizeof(wc); wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc = &editorProc; wc.hInstance = gModuleInstance;
        wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
        wc.hbrBackground = reinterpret_cast<HBRUSH>(static_cast<INT_PTR>(COLOR_WINDOW + 1));
        wc.lpszClassName = kEditorClassName;
        if (RegisterClassExW(&wc) != 0) return true;
        return GetLastError() == ERROR_CLASS_ALREADY_EXISTS;
    }
    bool openEditor(HWND parent) noexcept {
        if (parent == nullptr || !IsWindow(parent)) return false;
        HWND old = editorWindow_.load(std::memory_order_acquire);
        if (old != nullptr && IsWindow(old)) return true;
        if (!registerEditorClass()) return false;
        HWND w = CreateWindowExW(0, kEditorClassName, L"MUSE Perceptual Focus v1.3.2",
            WS_CHILD | WS_VISIBLE | WS_CLIPCHILDREN | WS_CLIPSIBLINGS,
            0, 0, kEditorWidth, kEditorHeight, parent, nullptr, gModuleInstance, this);
        if (w == nullptr) return false;
        editorWindow_.store(w, std::memory_order_release);
        ShowWindow(w, SW_SHOW); UpdateWindow(w); return true;
    }
    void closeEditor() noexcept {
        HWND w = editorWindow_.exchange(nullptr, std::memory_order_acq_rel);
        if (w != nullptr && IsWindow(w)) DestroyWindow(w);
        sliders_.fill(nullptr); valueLabels_.fill(nullptr);
        editorRefreshPending_.store(false, std::memory_order_release);
    }
    void requestEditorRefresh() noexcept {
        HWND w = editorWindow_.load(std::memory_order_acquire);
        if (w == nullptr) return;
        // Coalesce host parameter notifications: at most one refresh is queued.
        bool expected = false;
        if (!editorRefreshPending_.compare_exchange_strong(expected, true,
                std::memory_order_acq_rel, std::memory_order_relaxed)) return;
        if (!PostMessageW(w, kEditorRefreshMessage, 0, 0))
            editorRefreshPending_.store(false, std::memory_order_release);
    }
    void formatValue(std::size_t i, wchar_t* dst, std::size_t cap) const noexcept {
        if (dst == nullptr || cap == 0 || i >= kParameterCount) return;
        const double n = clamp(static_cast<double>(parameters_[i].load(std::memory_order_acquire)), 0.0, 1.0);
        switch (i) {
        case 3: std::swprintf(dst, cap, L"%.1f dB", n * 9.0); break;
        case 10: std::swprintf(dst, cap, L"%.2f Bark", denormalize(n, 0.35, 4.0)); break;
        case 12: std::swprintf(dst, cap, L"%d bands", resolutionBandCount(normalizedToResolution(static_cast<float>(n)))); break;
        case 14: std::swprintf(dst, cap, L"%.0f Hz", denormalize(n, 250.0, 1500.0)); break;
        default: std::swprintf(dst, cap, L"%.1f %%", n * 100.0); break;
        }
        dst[cap - 1] = L'\0';
    }
    void formatParameter(int32_t i, void* dst) const noexcept {
        if (dst == nullptr || i < 0 || static_cast<std::size_t>(i) >= kParameterCount) return;
        char text[VST_BUFFER_SIZE_PARAM_VALUE]{};
        const double n = clamp(static_cast<double>(parameters_[static_cast<std::size_t>(i)].load(std::memory_order_acquire)), 0.0, 1.0);
        switch (i) {
        case 3: std::snprintf(text, sizeof(text), "%.1f", n * 9.0); break;
        case 10: std::snprintf(text, sizeof(text), "%.2f", denormalize(n, 0.35, 4.0)); break;
        case 12: std::snprintf(text, sizeof(text), "%d", resolutionBandCount(normalizedToResolution(static_cast<float>(n)))); break;
        case 14: std::snprintf(text, sizeof(text), "%.0f", denormalize(n, 250.0, 1500.0)); break;
        default: std::snprintf(text, sizeof(text), "%.1f", n * 100.0); break;
        }
        copyVstString(dst, VST_BUFFER_SIZE_PARAM_VALUE, text);
    }
    bool createControls(HWND w) noexcept {
        const DWORD style = WS_CHILD | WS_VISIBLE;
        CreateWindowExW(0, L"STATIC", L"32 / 64 / 128 / 256 / 512 ERB regions | independent L/R | AVX1 | detector 250-1500 Hz",
            style, 18, 10, 900, 22, w, nullptr, gModuleInstance, nullptr);
        CreateWindowExW(0, L"BUTTON", L"Presets...", style | WS_TABSTOP | BS_PUSHBUTTON,
            946, 7, 150, 28, w,
            reinterpret_cast<HMENU>(static_cast<INT_PTR>(kEditorPresetButtonId)),
            gModuleInstance, nullptr);
        CreateWindowExW(0, L"STATIC", L"MASKING / CUT CONTROL",
            style, 18, 48, 500, 20, w, nullptr, gModuleInstance, nullptr);
        CreateWindowExW(0, L"STATIC", L"PREFERRED REGIONS / RESOLUTION / STEREO",
            style, 568, 48, 500, 20, w, nullptr, gModuleInstance, nullptr);
        for (std::size_t i = 0; i < kParameterCount; ++i) {
            const bool right = i >= kParametersPerColumn;
            const int x = right ? 568 : 18;
            const int y = editorRowY(i);
            CreateWindowExW(0, L"STATIC", kParameterNamesWide[i], style,
                x, y + 5, 142, 22, w, nullptr, gModuleInstance, nullptr);
            HWND slider = CreateWindowExW(0, TRACKBAR_CLASSW, L"",
                WS_CHILD | WS_VISIBLE | WS_TABSTOP | TBS_HORZ | TBS_NOTICKS,
                x + 146, y, 300, 32, w,
                reinterpret_cast<HMENU>(static_cast<INT_PTR>(kEditorSliderIdBase + static_cast<int>(i))),
                gModuleInstance, nullptr);
            if (slider == nullptr) return false;
            sliders_[i] = slider;
            SendMessageW(slider, TBM_SETRANGE, TRUE, MAKELPARAM(0, 1000));
            SendMessageW(slider, TBM_SETPAGESIZE, 0, 10);
            if (i == 12) SendMessageW(slider, TBM_SETTICFREQ, 500, 0);
            HWND value = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"", style | WS_TABSTOP | ES_RIGHT | ES_AUTOHSCROLL,
                x + 452, y + 3, 96, 25, w,
                reinterpret_cast<HMENU>(static_cast<INT_PTR>(kEditorSliderIdBase + 100 + static_cast<int>(i))),
                gModuleInstance, nullptr);
            if (value == nullptr) return false;
            valueLabels_[i] = value;
        }
        syncControls(); return true;
    }
    void syncControls() noexcept {
        for (std::size_t i = 0; i < kParameterCount; ++i) {
            if (sliders_[i] != nullptr) {
                const float v = clamp01f(parameters_[i].load(std::memory_order_acquire));
                const int pos = static_cast<int>(std::lround(v * 1000.0f));
                if (static_cast<int>(SendMessageW(sliders_[i], TBM_GETPOS, 0, 0)) != pos)
                    SendMessageW(sliders_[i], TBM_SETPOS, TRUE, static_cast<LPARAM>(pos));
            }
            if (valueLabels_[i] != nullptr && GetFocus() != valueLabels_[i]) {
                wchar_t text[64]{}; wchar_t currentText[64]{};
                formatValue(i, text, 64);
                GetWindowTextW(valueLabels_[i], currentText, 64);
                if (std::wcscmp(currentText, text) != 0) SetWindowTextW(valueLabels_[i], text);
            }
        }
    }
    void setFromEditor(std::size_t i, int position) noexcept {
        if (i >= kParameterCount) return;
        float v = static_cast<float>(std::clamp(position, 0, 1000)) / 1000.0f;
        if (i == 12) {
            const int mode = normalizedToResolution(v);
            v = resolutionToNormalized(mode);
            if (sliders_[i] != nullptr)
                SendMessageW(sliders_[i], TBM_SETPOS, TRUE, static_cast<LPARAM>(std::lround(v * 1000.0f)));
        }
        parameters_[i].store(v, std::memory_order_release);
        if (host_ != nullptr) host_(&effect_, 0, static_cast<int32_t>(i), 0, nullptr, v);
        if (valueLabels_[i] != nullptr) {
            wchar_t text[64]{}; formatValue(i, text, 64); SetWindowTextW(valueLabels_[i], text);
        }
    }
    void setFromEditorValue(std::size_t i, double value) noexcept {
        if (i >= kParameterCount || !std::isfinite(value)) return;
        float normalized = 0.0f;
        if (i == 3) normalized = normalizeLinear(std::clamp(value, 0.0, 9.0), 0.0, 9.0);
        else if (i == 10) normalized = normalizeLinear(std::clamp(value, 0.35, 4.0), 0.35, 4.0);
        else if (i == 12) {
            const int mode = resolutionModeFromBandCount(value);
            normalized = resolutionToNormalized(mode);
        } else if (i == 14) normalized = normalizeLinear(std::clamp(value, 250.0, 1500.0), 250.0, 1500.0);
        else normalized = normalizeLinear(std::clamp(value, 0.0, 100.0), 0.0, 100.0);
        parameters_[i].store(normalized, std::memory_order_release);
        if (i == 12 && sliders_[i] != nullptr)
            SendMessageW(sliders_[i], TBM_SETPOS, TRUE,
                static_cast<LPARAM>(std::lround(normalized * 1000.0f)));
        if (host_ != nullptr) host_(&effect_, 0, static_cast<int32_t>(i), 0, nullptr, normalized);
        syncControls();
    }

    void commitEditorValue(std::size_t i) noexcept {
        if (i >= kParameterCount || valueLabels_[i] == nullptr) return;
        wchar_t buffer[64]{};
        GetWindowTextW(valueLabels_[i], buffer, 64);
        for (wchar_t* c = buffer; *c != L'\0'; ++c) if (*c == L',') *c = L'.';
        wchar_t* end = nullptr;
        const double value = std::wcstod(buffer, &end);
        if (end == buffer || !std::isfinite(value)) {
            syncControls();
            return;
        }
        setFromEditorValue(i, value);
    }

    void applyPreset(int preset) noexcept {
        // Presets are starting points; all controls remain freely editable.
        Parameters p;
        switch (preset) {
        case 1: // Natural Balance: subtle broad cleanup, low fatigue.
            p.amount = 0.38; p.sensitivity = 0.56; p.softness = 0.70;
            p.maximumReductionDb = 2.8; p.energyPreference = 0.84;
            p.body = 0.28; p.vocal = 0.62; p.presence = 0.46;
            p.articulation = 0.30; p.air = 0.12; p.focusWidthBark = 1.55;
            p.responsiveness = 0.62; p.resolutionMode = 2; p.detectorRateHz = 1250.0; break;
        case 2: // Vocal Center: preserve the core vocal band and trim maskers around it.
            p.amount = 0.56; p.sensitivity = 0.68; p.softness = 0.68;
            p.maximumReductionDb = 3.8; p.energyPreference = 0.88;
            p.body = 0.16; p.vocal = 0.96; p.presence = 0.46;
            p.articulation = 0.20; p.air = 0.03; p.focusWidthBark = 0.95;
            p.responsiveness = 0.78; p.resolutionMode = 3; p.detectorRateHz = 1250.0; break;
        case 3: // Speech Intelligibility: upper vocal formants and consonant detail.
            p.amount = 0.60; p.sensitivity = 0.72; p.softness = 0.54;
            p.maximumReductionDb = 4.2; p.energyPreference = 0.82;
            p.body = 0.05; p.vocal = 0.68; p.presence = 0.90;
            p.articulation = 0.86; p.air = 0.22; p.focusWidthBark = 1.05;
            p.responsiveness = 0.82; p.resolutionMode = 3; p.detectorRateHz = 1500.0; break;
        case 4: // Presence + Air: detail without a broad high shelf.
            p.amount = 0.46; p.sensitivity = 0.62; p.softness = 0.72;
            p.maximumReductionDb = 3.2; p.energyPreference = 0.84;
            p.body = 0.05; p.vocal = 0.22; p.presence = 0.82;
            p.articulation = 0.88; p.air = 0.90; p.focusWidthBark = 1.25;
            p.responsiveness = 0.70; p.resolutionMode = 3; p.detectorRateHz = 1250.0; break;
        case 5: // Warm Foundation: weight and lower-mid continuity.
            p.amount = 0.42; p.sensitivity = 0.58; p.softness = 0.70;
            p.maximumReductionDb = 3.0; p.energyPreference = 0.82;
            p.body = 0.92; p.vocal = 0.62; p.presence = 0.20;
            p.articulation = 0.10; p.air = 0.02; p.focusWidthBark = 1.75;
            p.responsiveness = 0.58; p.resolutionMode = 2; p.detectorRateHz = 1000.0; break;
        case 6: // Dense Mix: broad target map, higher resolution, moderated decision rate.
            p.amount = 0.66; p.sensitivity = 0.76; p.softness = 0.66;
            p.maximumReductionDb = 4.8; p.energyPreference = 0.66;
            p.body = 0.42; p.vocal = 0.58; p.presence = 0.62;
            p.articulation = 0.58; p.air = 0.28; p.focusWidthBark = 2.10;
            p.responsiveness = 0.84; p.resolutionMode = 4; p.detectorRateHz = 750.0; break;
        case 7: // Low CPU: conservative analysis and cadence.
            p.amount = 0.42; p.sensitivity = 0.58; p.softness = 0.62;
            p.maximumReductionDb = 3.5; p.energyPreference = 0.72;
            p.body = 0.22; p.vocal = 0.62; p.presence = 0.42;
            p.articulation = 0.25; p.air = 0.08; p.focusWidthBark = 1.35;
            p.responsiveness = 0.72; p.resolutionMode = 1; p.detectorRateHz = 750.0; break;
        case 8: // Baudelaire Choice: equal perceptual-region priorities, maximum analysis and responsiveness.
            p.amount = 1.0; p.sensitivity = 1.0; p.softness = 1.0;
            p.maximumReductionDb = 9.0; p.energyPreference = 0.50;
            p.body = 0.50; p.vocal = 0.50; p.presence = 0.50;
            p.articulation = 0.50; p.air = 0.50; p.focusWidthBark = 0.35;
            p.responsiveness = 1.0; p.resolutionMode = 4;
            p.stereoLink = 0.0; p.detectorRateHz = 1500.0; break;
        default: // Factory default: unchanged v1.0 parameter values.
            break;
        }
        const std::array<double, 15> values = {
            p.amount, p.sensitivity, p.softness, p.maximumReductionDb,
            p.energyPreference, p.body, p.vocal, p.presence, p.articulation,
            p.air, p.focusWidthBark, p.responsiveness,
            static_cast<double>(p.resolutionMode), p.stereoLink, p.detectorRateHz
        };
        for (std::size_t i = 0; i < values.size(); ++i) {
            float n = 0.0f;
            if (i == 3) n = normalizeLinear(values[i], 0.0, 9.0);
            else if (i == 10) n = normalizeLinear(values[i], 0.35, 4.0);
            else if (i == 12) n = resolutionToNormalized(static_cast<int>(values[i]));
            else if (i == 14) n = normalizeLinear(values[i], 250.0, 1500.0);
            else n = static_cast<float>(values[i]);
            parameters_[i].store(n, std::memory_order_release);
            if (host_ != nullptr) host_(&effect_, 0, static_cast<int32_t>(i), 0, nullptr, n);
        }
        syncControls();
    }

    void showPresetMenu(HWND owner) noexcept {
        HMENU menu = CreatePopupMenu();
        if (menu == nullptr) return;
        AppendMenuW(menu, MF_STRING, 1, L"Factory Default \u2014 64 bands");
        AppendMenuW(menu, MF_STRING, 2, L"Natural Balance \u2014 128 bands");
        AppendMenuW(menu, MF_STRING, 3, L"Vocal Center \u2014 256 bands");
        AppendMenuW(menu, MF_STRING, 4, L"Speech Intelligibility \u2014 256 bands");
        AppendMenuW(menu, MF_STRING, 5, L"Presence + Air \u2014 256 bands");
        AppendMenuW(menu, MF_STRING, 6, L"Warm Foundation \u2014 128 bands");
        AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
        AppendMenuW(menu, MF_STRING, 7, L"Dense Mix \u2014 512 bands / 750 Hz");
        AppendMenuW(menu, MF_STRING, 8, L"Low CPU \u2014 64 bands / 750 Hz");
        AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
        AppendMenuW(menu, MF_STRING, 9, L"Baudelaire Choice \u2014 512 bands / 1500 Hz");
        RECT r{}; GetWindowRect(GetDlgItem(owner, kEditorPresetButtonId), &r);
        const UINT chosen = TrackPopupMenu(menu, TPM_RETURNCMD | TPM_RIGHTBUTTON,
            r.left, r.bottom, 0, owner, nullptr);
        DestroyMenu(menu);
        if (chosen >= 1 && chosen <= 9) applyPreset(static_cast<int>(chosen) - 1);
    }

    void layoutControls(HWND w) noexcept {
        RECT r{}; if (!GetClientRect(w, &r)) return;
        const int colWidth = (r.right - r.left) / 2;
        for (std::size_t i = 0; i < kParameterCount; ++i) {
            const int x = (i >= kParametersPerColumn) ? colWidth + 18 : 18;
            const int y = editorRowY(i);
            if (sliders_[i] != nullptr)
                MoveWindow(sliders_[i], x + 146, y, std::max(160, colWidth - 250), 32, TRUE);
            if (valueLabels_[i] != nullptr)
                MoveWindow(valueLabels_[i], x + colWidth - 112, y + 5, 96, 22, TRUE);
        }
    }
    static LRESULT CALLBACK editorProc(HWND w, UINT msg, WPARAM wp, LPARAM lp) noexcept {
        MuseEsPerceptualPlugin* self = reinterpret_cast<MuseEsPerceptualPlugin*>(GetWindowLongPtrW(w, GWLP_USERDATA));
        if (msg == WM_NCCREATE) {
            const auto* cs = reinterpret_cast<const CREATESTRUCTW*>(lp);
            self = static_cast<MuseEsPerceptualPlugin*>(cs->lpCreateParams);
            SetWindowLongPtrW(w, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
            if (self != nullptr) self->editorWindow_.store(w, std::memory_order_release);
        }
        if (self == nullptr) return DefWindowProcW(w, msg, wp, lp);
        switch (msg) {
        case WM_CREATE: return self->createControls(w) ? 0 : -1;
        case WM_SIZE: self->layoutControls(w); return 0;
        case WM_HSCROLL: {
            HWND slider = reinterpret_cast<HWND>(lp);
            if (slider != nullptr) {
                const int index = GetDlgCtrlID(slider) - kEditorSliderIdBase;
                if (index >= 0 && static_cast<std::size_t>(index) < kParameterCount)
                    self->setFromEditor(static_cast<std::size_t>(index),
                        static_cast<int>(SendMessageW(slider, TBM_GETPOS, 0, 0)));
            }
            return 0;
        }
        case WM_COMMAND: {
            const int id = LOWORD(wp);
            const int notification = HIWORD(wp);
            if (id == kEditorPresetButtonId && notification == BN_CLICKED) {
                self->showPresetMenu(w); return 0;
            }
            if (notification == EN_KILLFOCUS && id >= kEditorSliderIdBase + 100 &&
                id < kEditorSliderIdBase + 100 + static_cast<int>(kParameterCount)) {
                self->commitEditorValue(static_cast<std::size_t>(id - kEditorSliderIdBase - 100));
                return 0;
            }
            break;
        }
        case kEditorRefreshMessage:
            self->editorRefreshPending_.store(false, std::memory_order_release);
            self->syncControls();
            return 0;
        case WM_NCDESTROY:
            self->editorWindow_.store(nullptr, std::memory_order_release);
            self->editorRefreshPending_.store(false, std::memory_order_release);
            self->sliders_.fill(nullptr); self->valueLabels_.fill(nullptr);
            SetWindowLongPtrW(w, GWLP_USERDATA, 0);
            return DefWindowProcW(w, msg, wp, lp);
        default: return DefWindowProcW(w, msg, wp, lp);
        }
    }

    static intptr_t VST_FUNCTION_INTERFACE dispatch(vst_effect_t* e, int32_t opcode,
        int32_t index, intptr_t value, void* pointer, float opt) noexcept {
        auto* self = fromEffect(e); if (self == nullptr) return 0;
        switch (opcode) {
        case kVstEffectOpen: return 1;
        case kVstEffectClose: delete self; return 1;
        case kVstEffectSetProgram: return value == 0 ? 1 : 0;
        case kVstEffectGetProgram: return 0;
        case kVstEffectSetProgramName:
            if (pointer != nullptr) copyVstString(self->programName_, sizeof(self->programName_), static_cast<const char*>(pointer)); return 1;
        case kVstEffectGetProgramName: copyVstString(pointer, VST_BUFFER_SIZE_PROGRAM_NAME, self->programName_); return 1;
        case kVstEffectGetParamLabel:
            if (index < 0 || static_cast<std::size_t>(index) >= kParameterCount) return 0;
            copyVstString(pointer, VST_BUFFER_SIZE_PARAM_LABEL, kParameterLabels[static_cast<std::size_t>(index)]); return 1;
        case kVstEffectGetParamDisplay: self->formatParameter(index, pointer); return 1;
        case kVstEffectGetParamName:
            if (index < 0 || static_cast<std::size_t>(index) >= kParameterCount) return 0;
            copyVstString(pointer, VST_BUFFER_SIZE_PARAM_LONG_NAME, kParameterNames[static_cast<std::size_t>(index)]); return 1;
        case kVstEffectSetSampleRate:
            if (std::isfinite(opt) && opt >= 8000.0f && opt <= 768000.0f) {
                self->sampleRate_.store(static_cast<double>(opt), std::memory_order_release);
                // VST SetSampleRate is a host lifecycle callback. Rebuild here,
                // outside processBlock(), so the first audio block is already ready.
                if (!self->prepareForCurrentRate()) return 0;
            }
            return 1;
        case kVstEffectSetBlockSize: self->blockSize_ = static_cast<int32_t>(value); return 1;
        case kVstEffectMainsChanged:
            if (value == 0) {
                self->mainsActive_.store(false, std::memory_order_release);
                return 1;
            }
            // Defensive fallback for hosts that omit SetSampleRate or only
            // notify the format when restarting the effect.
            if (!self->isPreparedForCurrentRate() && !self->prepareForCurrentRate()) {
                self->mainsActive_.store(false, std::memory_order_release);
                return 0;
            }
            self->mainsActive_.store(true, std::memory_order_release);
            return 1;
        case kVstEffectEditGetRect:
            if (pointer == nullptr) return 0; *static_cast<vst_rect_t**>(pointer) = &editorRect_; return 1;
        case kVstEffectEditOpen: return self->openEditor(static_cast<HWND>(pointer)) ? 1 : 0;
        case kVstEffectEditClose: self->closeEditor(); return 0;
        case kVstEffectEditIdle: return 0; // No polling: host parameter changes refresh the editor on demand.
        case kVstEffectGetCategory: return VST_EFFECT_CATEGORY_EFFECT;
        case kVstEffectSetBypass: self->bypass_.store(value != 0, std::memory_order_release); return 1;
        case kVstEffectGetEffectName: copyVstString(pointer, VST_BUFFER_SIZE_EFFECT_NAME, "MUSE Perceptual Focus"); return 1;
        case kVstEffectGetVendorString: copyVstString(pointer, VST_BUFFER_SIZE_VENDOR_NAME, "MUSE Project"); return 1;
        case kVstEffectGetProductString: copyVstString(pointer, VST_BUFFER_SIZE_PRODUCT_NAME, "MUSE Perceptual Focus"); return 1;
        case kVstEffectGetVendorVersion: return kPluginVersion;
        case kVstEffectCanDo:
            if (pointer == nullptr) return -1;
            if (std::strcmp(static_cast<const char*>(pointer), "plugAsChannelInsert") == 0 ||
                std::strcmp(static_cast<const char*>(pointer), "plugAsSend") == 0 ||
                std::strcmp(static_cast<const char*>(pointer), "2in2out") == 0) return 1;
            return -1;
        case kVstEffectGetVstVersion: return kVstVersion24;
        default: return 0;
        }
    }
    static void VST_FUNCTION_INTERFACE setParameter(vst_effect_t* e, uint32_t i, float v) noexcept {
        auto* self = fromEffect(e); if (self == nullptr || i >= kParameterCount) return;
        v = clamp01f(v);
        if (i == 12) v = resolutionToNormalized(normalizedToResolution(v));
        const float previous = self->parameters_[i].load(std::memory_order_relaxed);
        if (std::abs(previous - v) <= 1.0e-6f) return;
        self->parameters_[i].store(v, std::memory_order_release);
        self->requestEditorRefresh();
    }
    static float VST_FUNCTION_INTERFACE getParameter(vst_effect_t* e, uint32_t i) noexcept {
        auto* self = fromEffect(e); if (self == nullptr || i >= kParameterCount) return 0.0f;
        return clamp01f(self->parameters_[i].load(std::memory_order_acquire));
    }
    static void processBlock(MuseEsPerceptualPlugin* self, const float* const* inputs,
        float** outputs, int32_t samples, bool accumulate) noexcept {
        if (self == nullptr || outputs == nullptr || samples <= 0) return;
        const float* inL = inputs != nullptr ? inputs[0] : nullptr;
        const float* inR = inputs != nullptr ? inputs[1] : nullptr;
        float* outL = outputs[0]; float* outR = outputs[1];
        if (outL == nullptr || outR == nullptr) return;
        const bool ready = self->isPreparedForCurrentRate();
        const bool bypass = self->bypass_.load(std::memory_order_acquire) ||
            !self->mainsActive_.load(std::memory_order_acquire) || !ready;
        if (bypass) {
            for (int32_t i = 0; i < samples; ++i) {
                const float l = inL != nullptr ? inL[i] : 0.0f;
                const float r = inR != nullptr ? inR[i] : 0.0f;
                if (accumulate) { outL[i] += l; outR[i] += r; }
                else { outL[i] = l; outR[i] = r; }
            }
            return;
        }
        // The host may invoke processBlock on any of its audio threads. Protect
        // the synchronous/fallback path as well, then restore the host's MXCSR.
        ScopedDenormalMode denormalMode;
        self->engine_.setParameters(self->makeParameters());
        const auto submit = self->parallelWorkerReady_ ? &submitChannel : nullptr;
        const auto wait = self->parallelWorkerReady_ ? &waitChannels : nullptr;
        self->engine_.processStereoBlock(inL, inR, outL, outR,
            static_cast<std::size_t>(samples), accumulate, submit, wait, self);
    }
    static void VST_FUNCTION_INTERFACE processReplacing(vst_effect_t* e,
        const float* const* inputs, float** outputs, int32_t samples) noexcept {
        processBlock(fromEffect(e), inputs, outputs, samples, false);
    }
    static void VST_FUNCTION_INTERFACE processAccumulating(vst_effect_t* e,
        const float* const* inputs, float** outputs, int32_t samples) noexcept {
        processBlock(fromEffect(e), inputs, outputs, samples, true);
    }
    bool isPreparedForCurrentRate() const noexcept {
        if (!prepared_.load(std::memory_order_acquire)) return false;
        const double sr = sampleRate_.load(std::memory_order_acquire);
        const double readyRate = preparedRate_.load(std::memory_order_acquire);
        return std::isfinite(sr) && std::abs(sr - readyRate) < 0.5;
    }
    bool prepareForCurrentRate() noexcept {
        const double sr = sampleRate_.load(std::memory_order_acquire);
        prepared_.store(false, std::memory_order_release);
        const bool ok = engine_.prepare(sr);
        if (ok) {
            engine_.setParameters(makeParameters());
            preparedRate_.store(sr, std::memory_order_release);
            prepared_.store(true, std::memory_order_release);
        }
        return ok;
    }

    vst_effect_t effect_{};
    vst_host_callback_t host_ = nullptr;
    StereoEngine engine_{};
    std::array<std::atomic<float>, kParameterCount> parameters_{};
    std::atomic<double> sampleRate_{44100.0};
    std::atomic<bool> bypass_{false};
    std::atomic<bool> mainsActive_{true};
    std::atomic<HWND> editorWindow_{nullptr};
    std::atomic<bool> editorRefreshPending_{false};
    std::array<HWND, kParameterCount> sliders_{};
    std::array<HWND, kParameterCount> valueLabels_{};
    std::array<ChannelWorker, 2> workers_{};
    bool parallelWorkerReady_ = false;
    inline static vst_rect_t editorRect_{0, 0, kEditorHeight, kEditorWidth};
    int32_t blockSize_ = VST_DEFAULT_BLOCK_SIZE;
    std::atomic<double> preparedRate_{0.0};
    std::atomic<bool> prepared_{false};
    char programName_[VST_BUFFER_SIZE_PROGRAM_NAME] = "Default";
};
} // namespace

extern "C" BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        gModuleInstance = instance;
        DisableThreadLibraryCalls(instance);
    }
    return TRUE;
}

extern "C" __declspec(dllexport) vst_effect_t* VST_FUNCTION_INTERFACE
VSTPluginMain(vst_host_callback_t callback) {
    if (callback == nullptr) return nullptr;
    const intptr_t hostVersion = callback(nullptr, kVstHostGetVstVersion, 0, 0, nullptr, 0.0f);
    if (hostVersion <= 0) return nullptr;
    auto* plugin = new (std::nothrow) MuseEsPerceptualPlugin(callback);
    return plugin != nullptr ? plugin->effect() : nullptr;
}

#endif // MUSE_SELF_TEST

#endif // MUSE_AVX_KERNEL

