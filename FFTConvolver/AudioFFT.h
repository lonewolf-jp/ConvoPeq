// ==================================================================================
// Copyright (c) 2017 HiFi-LoFi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is furnished
// to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ==================================================================================

#ifndef _AUDIOFFT_H
#define _AUDIOFFT_H

#include <cstddef>

namespace audiofft
{

#ifdef FFTCONVOLVER_USE_DOUBLE
  typedef double Sample;
#else
  typedef float Sample;
#endif

  namespace detail
  {
    class AudioFFTImpl;
  }

  /**
   * @class AudioFFT
   * @brief FFT implementation
   */
  class AudioFFT
  {
  public:
    AudioFFT();
    ~AudioFFT();

    /**
     * @brief Initializes the FFT
     * @param size The size of the FFT (must be a power of 2)
     */
    void init(size_t size);

    /**
     * @brief Performs the forward FFT
     * @param data The input data (time domain)
     * @param re The real part of the output data (frequency domain)
     * @param im The imaginary part of the output data (frequency domain)
     */
    void fft(const Sample* data, Sample* re, Sample* im);

    /**
     * @brief Performs the inverse FFT
     * @param data The output data (time domain)
     * @param re The real part of the input data (frequency domain)
     * @param im The imaginary part of the input data (frequency domain)
     */
    void ifft(Sample* data, const Sample* re, const Sample* im);

    static size_t ComplexSize(size_t size);

  private:
    detail::AudioFFTImpl* _impl;
  };

} // End of namespace

#endif // Header guard