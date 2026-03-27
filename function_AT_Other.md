# [ATT]→[ATT以外]データ連携関数リスト

このファイルは、[ATT]（Audio Thread）に分類されている関数のうち、[ATT]以外の関数・データとやり取りしている関数をソースコードファイルごとにリストアップしたものです。

---

## EQProcessor.cpp

- EQProcessor::process(juce::dsp::AudioBlock<double>& block)
  - [MGT]のsetBandFrequency/setBandGain/setBandQ/setBandEnabled/setTotalGain/setAGCEnabled等で設定されたパラメータを参照
  - [MGT]のgetBandParams/getEQState等のデータを間接的に利用

- EQProcessor::prepareToPlay(double sampleRate, int newMaxInternalBlockSize)
  - [MGT]のパラメータ初期化・状態同期と連携

---

## Fixed15TapNoiseShaper.h

- processStereoBlock, processSample
  - [MGT]のsetCoefficients/prepare/reset等でセットされたデータを参照

---

## FixedNoiseShaper.h

- processStereoBlock, processSample
  - [MGT]のsetCoefficients/prepare/reset等でセットされたデータを参照

---

## ConvolverProcessor.h

- prepareToPlay, releaseResources, process
  - [MGT]のsetBypass/setMix/setPhaseMode等でセットされたパラメータを参照
  - [WLT]のloadImpulseResponseでロードされたIRデータを利用

---

## AudioEngineProcessor.cpp

- prepareToPlay, releaseResources, processBlock
  - [MGT]のAudioEngineProcessor::setCurrentProgram等でセットされた状態を参照

---

## AudioEngine.h / AudioEngine.cpp

- prepareToPlay, releaseResources, getNextAudioBlock, processBlockDouble
  - [MGT]のsetEqBypassRequested/setConvolverBypassRequested/setConvolverPhaseMode等でセットされたパラメータを参照
  - [WLT]のstartNoiseShaperLearning/stopNoiseShaperLearning等で学習済みデータを参照

---

## InputBitDepthTransform.h

- これらの[ATT]関数は主に外部から与えられたデータを変換するが、[ATT]以外の関数との直接的なやり取りは少ない（主にデータフロー上の連携）

---

## LatticeNoiseShaper.h

- processStereoBlock
  - [MGT]のsetCoefficients/prepare/reset等でセットされたデータを参照

---

## LockFreeRingBuffer.h

- push, pop, size, clear
  - [ATT]以外のスレッド（UI/Worker等）とロックフリーでデータをやり取りするため、[ATT]以外との明確な連携あり

---

※ 各関数の詳細なやり取り内容は、パラメータ・メンバ変数・バッファ等の共有を通じて行われています。
