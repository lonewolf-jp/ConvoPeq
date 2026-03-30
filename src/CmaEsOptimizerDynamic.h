#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>

//==============================================================================
/**
    CmaEsOptimizerDynamic: 可変次元対応の簡易 CMA-ES オプティマイザ
    Phase 3 の全通過フィルタ最適化の基盤となる。
*/
class CmaEsOptimizerDynamic {
public:
    struct Params {
        double sigmaMin = 0.03;
        double sigmaMax = 0.30;
        double covRetentionTarget = 0.92;
        double covRetentionStep = 0.0;
    };

    explicit CmaEsOptimizerDynamic(int dimension);
    ~CmaEsOptimizerDynamic() = default;

    void setParams(const Params& p) noexcept { params = p; }
    void setSeed(uint64_t seed) { rng.seed(static_cast<std::mt19937::result_type>(seed)); }
    void initFromParcor(const double* initialMean);
    void sample(std::vector<std::vector<double>>& candidates);
    void update(const std::vector<std::vector<double>>& candidates,
                const std::vector<double>& fitness);
    void getMean(double* outMean) const { std::copy(mean.begin(), mean.end(), outMean); }
    void serializeTo(double* outMean, double* outCov, double& outSigma) const;
    void deserializeFrom(const double* inMean, const double* inCov, double inSigma);

private:
    int dim;
    std::vector<double> mean;
    std::vector<double> covariance;  // dim * dim
    double sigma;
    double covRetentionCurrent;
    Params params;
    std::mt19937 rng;

    void resetIdentityCovariance();
    void computeCholesky(std::vector<double>& lowerTriangular) const;
    static double sanitize(double x) { return (std::abs(x) < 1e-15) ? 0.0 : x; }
};
