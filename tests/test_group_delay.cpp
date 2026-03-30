#include "../src/AllpassDesigner.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double phaseResponse(double rho, double theta, double omega) {
    double phase1 = std::atan2(rho * std::sin(omega - theta), 1.0 - rho * std::cos(omega - theta));
    double phase2 = std::atan2(rho * std::sin(omega + theta), 1.0 - rho * std::cos(omega + theta));
    return -2.0 * omega - 2.0 * (phase1 + phase2);
}

static double unwrapPhase(double prev, double curr) {
    double diff = curr - prev;
    if (diff > M_PI) return curr - 2.0 * M_PI;
    if (diff < -M_PI) return curr + 2.0 * M_PI;
    return curr;
}

double numericGroupDelay(double rho, double theta, double omega, double eps = 1e-6) {
    double phi_p = phaseResponse(rho, theta, omega + eps);
    double phi_m = phaseResponse(rho, theta, omega - eps);
    double phi_c = phaseResponse(rho, theta, omega);
    phi_p = unwrapPhase(phi_c, phi_p);
    phi_m = unwrapPhase(phi_c, phi_m);
    return -(phi_p - phi_m) / (2.0 * eps);
}

int main() {
    const double rho = 0.7;
    const double theta = M_PI / 3.0;
    const std::vector<double> test_omega = {0.01, 0.1, 0.5, 1.0, 2.0, 3.0};
    double max_error = 0.0;
    
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Testing Group Delay Formula Accuracy..." << std::endl;
    
    for (double w : test_omega) {
        double analytic = convo::AllpassDesigner::sectionGroupDelayRhoTheta(rho, theta, w, 48000.0);
        double numeric = numericGroupDelay(rho, theta, w);
        double error = std::abs(analytic - numeric);
        max_error = std::max(max_error, error);
        std::cout << "omega=" << w << " analytic=" << analytic << " numeric=" << numeric << " diff=" << error << std::endl;
    }
    
    std::cout << "Max error: " << max_error << std::endl;
    
    if (max_error < 1e-8) {
        std::cout << "SUCCESS: Analytic formula matches numeric differentiation." << std::endl;
        return 0;
    } else {
        std::cout << "FAILURE: Error is too large!" << std::endl;
        return 1;
    }
}
