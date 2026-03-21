#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <mkl.h>

class CmaEsOptimizer
{
public:
    static constexpr int kDim = 9;
    static constexpr int kPopulation = 18;
    static constexpr int kElite = 6;

    struct Params {
        double sigmaMin     = 0.03;
        double sigmaMax     = 0.30;
        double covRetentionTarget = 0.92;
        double covRetentionStep   = 0.0;
    };

    CmaEsOptimizer()
    {
        mean = static_cast<double*>(mkl_malloc(kDim * sizeof(double), 64));
        covariance = static_cast<double*>(mkl_malloc(kDim * kDim * sizeof(double), 64));

        std::random_device device;
        rng.seed(device());
        
        for (int i = 0; i < kDim; ++i) mean[i] = 0.0;
        resetIdentityCovariance();
    }

    ~CmaEsOptimizer()
    {
        mkl_free(mean);
        mkl_free(covariance);
    }

    void setParams(const Params& p) noexcept
    {
        params = p;
    }

    void serializeCovUpperTriangle(double* out45) const noexcept
    {
        int idx = 0;
        for (int r = 0; r < kDim; ++r)
            for (int c = r; c < kDim; ++c)
                out45[idx++] = covariance[r * kDim + c];
    }

    void deserializeCovUpperTriangle(const double* in45) noexcept
    {
        int idx = 0;
        for (int r = 0; r < kDim; ++r)
            for (int c = r; c < kDim; ++c)
            {
                covariance[r * kDim + c] = in45[idx];
                covariance[c * kDim + r] = in45[idx];
                ++idx;
            }
    }

    void serializeTo(double* outMean9, double* outCov45, double& outSigma) const noexcept
    {
        for (int i = 0; i < kDim; ++i) outMean9[i] = mean[i];
        serializeCovUpperTriangle(outCov45);
        outSigma = sigma;
    }

    void deserializeFrom(const double* inMean9, const double* inCov45, double inSigma) noexcept
    {
        for (int i = 0; i < kDim; ++i) mean[i] = inMean9[i];
        deserializeCovUpperTriangle(inCov45);
        sigma = inSigma;
    }

    void initFromParcor(const double* initialParcor) noexcept
    {
        for (int i = 0; i < kDim; ++i)
            mean[i] = parcorToUnconstrained(initialParcor[i]);

        sigma = 0.12;
        covRetentionCurrent = params.covRetentionTarget;
        resetIdentityCovariance();
    }

    void sample(double candidates[kPopulation][kDim])
    {
        double lowerTriangular[kDim][kDim] = {};
        computeCholesky(lowerTriangular);

        std::normal_distribution<double> normalDist(0.0, 1.0);

        for (int populationIndex = 0; populationIndex < kPopulation; ++populationIndex)
        {
            double z[kDim] = {};

            for (int dim = 0; dim < kDim; ++dim)
                z[dim] = normalDist(rng);

            for (int dim = 0; dim < kDim; ++dim)
            {
                double correlated = 0.0;
                for (int column = 0; column <= dim; ++column)
                    correlated += lowerTriangular[dim][column] * z[column];

                candidates[populationIndex][dim] = sanitize(mean[dim] + sigma * correlated);
            }
        }
    }

    void update(const double candidates[kPopulation][kDim], const double fitness[kPopulation]) noexcept
    {
        covRetentionCurrent = std::min(params.covRetentionTarget,
                                       covRetentionCurrent + params.covRetentionStep);

        int sortedIndices[kPopulation] = {};
        for (int i = 0; i < kPopulation; ++i)
            sortedIndices[i] = i;

        std::sort(std::begin(sortedIndices), std::end(sortedIndices),
                  [&fitness](int lhs, int rhs) { return fitness[lhs] < fitness[rhs]; });

        double oldMean[kDim] = {};
        for (int dim = 0; dim < kDim; ++dim)
            oldMean[dim] = mean[dim];

        double newMean[kDim] = {};
        for (int eliteIndex = 0; eliteIndex < kElite; ++eliteIndex)
        {
            const int candidateIndex = sortedIndices[eliteIndex];
            for (int dim = 0; dim < kDim; ++dim)
                newMean[dim] += candidates[candidateIndex][dim];
        }

        for (int dim = 0; dim < kDim; ++dim)
            newMean[dim] /= static_cast<double>(kElite);

        for (int row = 0; row < kDim; ++row)
        {
            for (int column = 0; column < kDim; ++column)
            {
                double eliteCov = 0.0;
                for (int eliteIndex = 0; eliteIndex < kElite; ++eliteIndex)
                {
                    const int candidateIndex = sortedIndices[eliteIndex];
                    const double yRow = (candidates[candidateIndex][row] - oldMean[row]) / sigma;
                    const double yColumn = (candidates[candidateIndex][column] - oldMean[column]) / sigma;
                    eliteCov += yRow * yColumn;
                }
                covariance[row * kDim + column]
                    = sanitize((covRetentionCurrent * covariance[row * kDim + column])
                    + (((1.0 - covRetentionCurrent) / static_cast<double>(kElite)) * eliteCov));
            }
        }

        double variance = 0.0;
        for (int eliteIndex = 0; eliteIndex < kElite; ++eliteIndex)
        {
            const int candidateIndex = sortedIndices[eliteIndex];
            for (int row = 0; row < kDim; ++row)
            {
                const double deltaRow = candidates[candidateIndex][row] - newMean[row];
                variance += deltaRow * deltaRow;
            }
        }

        for (int dim = 0; dim < kDim; ++dim)
            mean[dim] = sanitize(newMean[dim]);

        sigma = std::clamp(std::sqrt(variance / static_cast<double>(kElite * kDim)), params.sigmaMin, params.sigmaMax);
    }

    static void toParcor(const double* unconstrained, double* parcor) noexcept
    {
        for (int i = 0; i < kDim; ++i)
            parcor[i] = sanitize(std::tanh(unconstrained[i]));
    }

    void getMeanParcor(double* outParcor) const noexcept
    {
        toParcor(mean, outParcor);
    }

private:
    static inline double sanitize(double x) noexcept
    {
        return (std::abs(x) < 1e-15) ? 0.0 : x;
    }

    static double parcorToUnconstrained(double value) noexcept
    {
        constexpr double kLimit = 0.995;
        const double clamped = std::clamp(value, -kLimit, kLimit);
        return 0.5 * std::log((1.0 + clamped) / (1.0 - clamped));
    }

    void resetIdentityCovariance() noexcept
    {
        for (int row = 0; row < kDim; ++row)
            for (int column = 0; column < kDim; ++column)
                covariance[row * kDim + column] = (row == column) ? 1.0 : 0.0;
    }

    void computeCholesky(double lowerTriangular[kDim][kDim]) const noexcept
    {
        for (int row = 0; row < kDim; ++row)
        {
            for (int column = 0; column <= row; ++column)
            {
                double sum = covariance[row * kDim + column];
                for (int k = 0; k < column; ++k)
                    sum -= lowerTriangular[row][k] * lowerTriangular[column][k];

                if (row == column)
                    lowerTriangular[row][column] = std::sqrt(std::max(sum, 1.0e-9));
                else
                    lowerTriangular[row][column] = sum / std::max(lowerTriangular[column][column], 1.0e-9);
            }
        }
    }

    double* mean = nullptr;
    double* covariance = nullptr;
    double sigma = 0.12;
    double covRetentionCurrent = 0.92;
    Params params;
    std::mt19937 rng;
};
