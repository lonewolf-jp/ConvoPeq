#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>

class CmaEsOptimizer
{
public:
    static constexpr int kDim = 9;
    static constexpr int kPopulation = 18;
    static constexpr int kElite = 6;

    CmaEsOptimizer()
    {
        std::random_device device;
        rng.seed(device());
        resetIdentityCovariance();
    }

    void initFromParcor(const double* initialParcor) noexcept
    {
        for (int i = 0; i < kDim; ++i)
            mean[static_cast<size_t>(i)] = parcorToUnconstrained(initialParcor[i]);

        sigma = 0.12;
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

                candidates[populationIndex][dim] = mean[static_cast<size_t>(dim)] + sigma * correlated;
            }
        }
    }

    void update(const double candidates[kPopulation][kDim], const double fitness[kPopulation]) noexcept
    {
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
                covariance[static_cast<size_t>(row)][static_cast<size_t>(column)]
                    = (0.92 * covariance[static_cast<size_t>(row)][static_cast<size_t>(column)])
                    + ((0.08 / static_cast<double>(kElite)) * eliteCov);
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
            mean[static_cast<size_t>(dim)] = newMean[dim];

        sigma = std::clamp(std::sqrt(variance / static_cast<double>(kElite * kDim)), 0.03, 0.30);
    }

    static void toParcor(const double* unconstrained, double* parcor) noexcept
    {
        for (int i = 0; i < kDim; ++i)
            parcor[i] = std::tanh(unconstrained[i]);
    }

    void getMeanParcor(double* outParcor) const noexcept
    {
        toParcor(mean.data(), outParcor);
    }

private:
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
                covariance[static_cast<size_t>(row)][static_cast<size_t>(column)] = (row == column) ? 1.0 : 0.0;
    }

    void computeCholesky(double lowerTriangular[kDim][kDim]) const noexcept
    {
        for (int row = 0; row < kDim; ++row)
        {
            for (int column = 0; column <= row; ++column)
            {
                double sum = covariance[static_cast<size_t>(row)][static_cast<size_t>(column)];
                for (int k = 0; k < column; ++k)
                    sum -= lowerTriangular[row][k] * lowerTriangular[column][k];

                if (row == column)
                    lowerTriangular[row][column] = std::sqrt(std::max(sum, 1.0e-9));
                else
                    lowerTriangular[row][column] = sum / std::max(lowerTriangular[column][column], 1.0e-9);
            }
        }
    }

    std::array<double, kDim> mean {};
    std::array<std::array<double, kDim>, kDim> covariance {};
    double sigma = 0.12;
    std::mt19937 rng;
};
