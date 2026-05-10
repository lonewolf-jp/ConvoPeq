#pragma once

namespace convo {

struct BuildInput {
    double sampleRate = 0.0;
    int blockSize = 0;
    int ditherBitDepth = 0;
    int oversamplingFactor = 0;
    int oversamplingType = 0;
    int noiseShaperType = 0;
};

} // namespace convo
