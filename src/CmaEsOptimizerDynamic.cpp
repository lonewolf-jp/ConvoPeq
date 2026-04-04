#include "CmaEsOptimizerDynamic.h"
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>

CmaEsOptimizerDynamic::CmaEsOptimizerDynamic(int dimension)
    : dim(dimension), sigma(0.12), covRetentionCurrent(0.92) {
    mean.resize(dim, 0.0);
    covariance.resize(dim * dim, 0.0);
    resetIdentityCovariance();
    std::random_device rd;
    rng.seed(rd());
}

void CmaEsOptimizerDynamic::resetIdentityCovariance() {
    std::fill(covariance.begin(), covariance.end(), 0.0);
    for (int i = 0; i < dim; ++i)
        covariance[i * dim + i] = 1.0;
}

void CmaEsOptimizerDynamic::computeCholesky(std::vector<double>& lowerTriangular) const {
    lowerTriangular.assign(dim * dim, 0.0);
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col <= row; ++col) {
            double sum = covariance[row * dim + col];
            for (int k = 0; k < col; ++k)
                sum -= lowerTriangular[row * dim + k] * lowerTriangular[col * dim + k];
            if (row == col)
                lowerTriangular[row * dim + col] = std::sqrt(std::max(sum, 1e-9));
            else
                lowerTriangular[row * dim + col] = sum / std::max(lowerTriangular[col * dim + col], 1e-9);
        }
    }
}

void CmaEsOptimizerDynamic::initFromParcor(const double* initialMean) {
    for (int i = 0; i < dim; ++i)
        mean[i] = initialMean[i];
    sigma = 0.12;
    covRetentionCurrent = params.covRetentionTarget;
    resetIdentityCovariance();
}

void CmaEsOptimizerDynamic::sample(std::vector<std::vector<double>>& candidates) {
    std::vector<double> lowerTriangular;
    computeCholesky(lowerTriangular);
    std::normal_distribution<double> normalDist(0.0, 1.0);

    for (auto& candidate : candidates) {
        std::vector<double> z(dim);
        for (int d = 0; d < dim; ++d) z[d] = normalDist(rng);
        for (int d = 0; d < dim; ++d) {
            double correlated = 0.0;
            for (int j = 0; j <= d; ++j)
                correlated += lowerTriangular[d * dim + j] * z[j];
            candidate[d] = sanitize(mean[d] + sigma * correlated);
        }
    }
}

void CmaEsOptimizerDynamic::update(const std::vector<std::vector<double>>& candidates,
                                   const std::vector<double>& fitness) {
    const int lambda = static_cast<int>(candidates.size());
    const int mu = lambda / 2;

    covRetentionCurrent = std::min(params.covRetentionTarget,
                                   covRetentionCurrent + params.covRetentionStep);

    // ランキング
    std::vector<int> indices(lambda);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return fitness[a] < fitness[b]; });

    // 重み（対数減少）
    std::vector<double> weights(mu);
    double sumWeights = 0.0;
    for (int i = 0; i < mu; ++i) {
        weights[i] = std::log(double(mu) + 0.5) - std::log(double(i) + 1.0);
        sumWeights += weights[i];
    }
    for (int i = 0; i < mu; ++i) weights[i] /= sumWeights;

    std::vector<double> oldMean = mean;
    std::fill(mean.begin(), mean.end(), 0.0);
    for (int i = 0; i < mu; ++i) {
        const auto& candidate = candidates[indices[i]];
        for (int d = 0; d < dim; ++d)
            mean[d] += weights[i] * candidate[d];
    }

    // 共分散更新
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            double eliteCov = 0.0;
            for (int i = 0; i < mu; ++i) {
                const auto& candidate = candidates[indices[i]];
                double yRow = (candidate[row] - oldMean[row]) / sigma;
                double yCol = (candidate[col] - oldMean[col]) / sigma;
                eliteCov += weights[i] * yRow * yCol;
            }
            covariance[row * dim + col] = sanitize(
                covRetentionCurrent * covariance[row * dim + col] +
                (1.0 - covRetentionCurrent) * eliteCov
            );
        }
    }

    // 共分散の正定値性を維持（対角に微小値を加算）
    for (int i = 0; i < dim; ++i)
        covariance[i * dim + i] += 1e-9;

    // step-size 適応（次元スケーリング対応版）
    // cs は Hansen (2016) 推奨の次元依存値。大きな dim でも安定して機能する。
    double stepNorm = 0.0;
    for (int d = 0; d < dim; ++d) {
        double diff = mean[d] - oldMean[d];
        stepNorm += diff * diff;
    }
    stepNorm = std::sqrt(stepNorm / static_cast<double>(dim));
    const double expectedStep = sigma * std::sqrt(static_cast<double>(dim));
    const double ratio = stepNorm / (expectedStep + 1e-12);
    const double cs = std::min(1.0, (2.0 + std::log(static_cast<double>(dim) + 1.0)) /
                                    (std::sqrt(static_cast<double>(dim)) + 10.0));
    sigma *= std::exp((cs / (1.0 - cs + 1e-12)) * (ratio - 1.0));
    sigma = std::clamp(sigma, params.sigmaMin, params.sigmaMax);
}

void CmaEsOptimizerDynamic::serializeTo(double* outMean, double* outCov, double& outSigma) const {
    if (outMean) std::copy(mean.begin(), mean.end(), outMean);
    if (outCov) {
        int idx = 0;
        for (int r = 0; r < dim; ++r)
            for (int c = r; c < dim; ++c)
                outCov[idx++] = covariance[static_cast<size_t>(r * dim + c)];
    }
    outSigma = sigma;
}

void CmaEsOptimizerDynamic::getCovarianceUpperTriangle(std::vector<double>& out) const {
    out.resize(static_cast<size_t>(dim * (dim + 1) / 2));
    int idx = 0;
    for (int r = 0; r < dim; ++r)
        for (int c = r; c < dim; ++c)
            out[static_cast<size_t>(idx++)] = covariance[static_cast<size_t>(r * dim + c)];
}

void CmaEsOptimizerDynamic::deserializeFrom(const double* inMean, const double* inCov, double inSigma) {
    std::copy(inMean, inMean + dim, mean.begin());
    int idx = 0;
    for (int r = 0; r < dim; ++r)
        for (int c = r; c < dim; ++c) {
            covariance[static_cast<size_t>(r * dim + c)] = inCov[idx];
            covariance[static_cast<size_t>(c * dim + r)] = inCov[idx];
            ++idx;
        }
    sigma = inSigma;
}

