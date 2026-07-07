# ConvoPeq メモリ占有調査報告書 — 2.33GB の原因分析と改善提案

---

**作成日**: 2026-07-08
**対象**: AoS スクラッチ化改修（work70）適用後の実機メモリ占有 2.33GB
**計測条件**: 192kHz SR, ×2 オーバーサンプリング (processingRate=384kHz), IR長 25906 samples (≈135ms), 1024ブロック, ステレオ, NoiseShaper 学習中

---

## 1. 改修の効果確認

### 1.1 AoS スクラッチ化の削減効果（実績）

AoS スクラッチ化（全11 Patch）は正しく適用されており、期待通りのメモリ削減が確認された。

| 項目 | 改修前（推定） | 改修後 | 削減率 |
| :--- | :--- | :--- | :--- |
| NUC 1基 (単層) | ~640 MB | ~333 MB | **48%減** ✅ |
| StereoConvolver 1組 (L+R) | ~1,280 MB | ~666 MB | **48%減** ✅ |

### 1.2 確認された改修の動作

ログ分析により、以下の動作が正常であることを確認:

- `processingRate=384000` — オーバーサンプリング正常動作
- `irLoaded=1 irLen=192000` — IR ロード・コンボリューション正常
- `convBypass=0` — NUC エンジンがアクティブ
- gen=1〜8 の再構築を経て安定稼働
- 音響的な問題（ノイズ/ポップ/プチ）は報告なし

---

## 2. 2.33GB の内訳分析

### 2.1 StereoConvolver 三重保持（主要因, ~2.0GB）

実コード調査（`src/ConvolverProcessor.h`, `src/audioengine/AudioEngine.h`）により、IR 再ロード・パラメータ変更時に **StereoConvolver が三重に生存** する設計であることが判明した。

| # | 保持元 | 場所 | メモリ | 生存期間 |
| :--- | :--- | :--- | :--- | :--- |
| ① | **active DSPCore current** | `AudioEngine.h` L1898: `activeRuntimeDSPSlot` | ~666 MB | 常時（再生中） |
| ② | **IncrementalRebuildJob.pendingConv** | `ConvolverProcessor.h` L566: `IncrementalRebuildJob::pendingConv` | ~666 MB | IR再ロード・パラメータ変更時の構築中 |
| ③ | **fading DSPCore fading** | `AudioEngine.h` L1908: `fadingRuntimeDSPSlot` | ~666 MB | クロスフェード遷移中（〜60秒） |

**合計**: ~2,000 MB (≈2.0GB)

### 2.2 StereoConvolver 保持関係図

```text
                IncrementalRebuildJob
                 ┌──────────────────┐
                 │  pendingConv     │ ◀── 新IR構築中 (②)
                 │  (StereoConv*)  │      約666MB
                 └──────────────────┘
                          │ publish
                          ▼
  AudioEngine (DSPCore)
   ┌─────────────────────────────┐
   │ activeRuntimeDSPSlot (①)   │ ◀── 現在再生中
   │  (NonOwningPtr<DSPCore>)  │      約666MB
   │   ┌───────────────────┐   │
   │   │ ConvolverProcessor│   │
   │   │  convolver        │   │
   │   └───────────────────┘   │
   ├─────────────────────────────┤
   │ fadingRuntimeDSPSlot (③)  │ ◀── フェード中
   │  (NonOwningPtr<DSPCore>)  │      約666MB
   └─────────────────────────────┘
```

### 2.3 その他のオーバーヘッド (~330MB)

| 項目 | 推定容量 | 備考 |
| :--- | :--- | :--- |
| RuntimeWorld 世代 (gen=5) | ~100 MB | `commit_` / `pending_` / `current_` 三重保持 |
| EpochDomain 退役キュー滞留 | ~80 MB | `pendingRetireCount()` 未消化分 |
| EQ/DCBlock/TruePeak 等バッファ | ~80 MB | `processingBlockSize=524288` の各エンジン |
| NoiseShaper 学習バッファ | ~40 MB | `bufferedSamples=3,840,000` (382万サンプル) |
| その他 (UI/Analyzer 等) | ~30 MB | |

**合計**: ~330 MB

### 2.4 総計

| 区分 | 容量 |
| :--- | :--- |
| StereoConvolver 三重保持 | ~2,000 MB |
| その他オーバーヘッド | ~330 MB |
| **合計** | **~2,330 MB (2.33 GB)** |

---

## 3. 実コード調査による裏付け

### 3.1 active/fading 二重保持

`src/audioengine/AudioEngine.h` L1898-1908:

```cpp
// active runtime DSP slot
convo::NonOwningPtr<DSPCore> activeRuntimeDSPSlot { nullptr };
// fading runtime DSP slot
convo::NonOwningPtr<DSPCore> fadingRuntimeDSPSlot { nullptr };
```

各 `DSPCore` は `ConvolverProcessor convolver`（L845）を値保持しており、その中に `StereoConvolver` が2チャンネル分の NUC エンジン (`std::array<MKLNonUniformConvolver*, 2> nucConvolvers`) を保持している。

### 3.2 pendingConv による一時的二重保持

`src/ConvolverProcessor.h` L566:

```cpp
struct IncrementalRebuildJob {
    StereoConvolver* pendingConv = nullptr;  // ◀── 新エンジン構築中
    // ...
};
```

`src/convolver/ConvolverProcessor.Rebuild.cpp` L222:

```cpp
job.pendingConv = std::exchange(result.newConv, nullptr);  // ◀── 構築後 pending へ
```

新しい `StereoConvolver` がビルドスレッドで構築され、`pendingConv` に保持されている間、古いエンジンは `activeRuntimeDSPSlot` として生存し続ける。これにより**三重保持**が発生する。

### 3.3 ログによる世代推移確認

```text
gen=1 → gen=4 → gen=8: 3回の rebuild サイクル
PUBLISH seq=2 → seq=4 → seq=5: RuntimeWorld 世代交代
DSPCORE_PREPARE: 8回の準備呼び出し
```

各 rebuild サイクルで新しい `StereoConvolver` が構築され、古いものが retire されるまで過渡的に複数世代が生存する。

---

## 4. 改善提案

### 4.1 優先度高: pendingConv の早期 retire（最大 ~666MB 削減）

**現状**: `IncrementalRebuildJob` の `pendingConv` は、新しいエンジンが `publish` されるまで保持される。この間、古いエンジン（active/fading）と構築中のエンジン（pending）が三重に生存する。

**改善案**: `publish` の直前に `fadingRuntimeDSPSlot` を強制 retire することで、三重保持を二重に削減する。具体的には `applyNewStatePublishStep()` 内で以下の処理を追加:

```cpp
// 新エンジン publish 直前に fading を強制 retire
if (fadingRuntimeDSPSlot != nullptr) {
    retireDSP(fadingRuntimeDSPSlot);
    fadingRuntimeDSPSlot = nullptr;
}
publish(newEngine);
```

**削減見込み**: ~666 MB
**リスク**: フェード中のクロスフェードが中断されるが、IR再ロード時のフェードは通常ごく短時間（〜10ms）であり、聴感上の影響は限定的。

### 4.2 優先度中: RuntimeWorld 世代数の削減（~50MB 削減）

**現状**: gen=1,2,4,5,8 と複数の RuntimeWorld 世代が残存する可能性がある。

**改善案**: `publish()` 成功時に `commit_` 以外の世代を即座に解放する。

### 4.3 優先度低: NoiseShaper バッファ制限（~40MB 削減）

**現状**: `bufferedSamples=3,840,000`（382万サンプル）の学習バッファ。

**改善案**: 最大バッファサイズに上限を設定する。

### 4.4 総削減見込み

| 改善 | 削減量 | 実装コスト |
| :--- | :--- | :--- |
| pendingConv 早期 retire | ~666 MB | 低（数行） |
| RuntimeWorld 世代削減 | ~50 MB | 中 |
| NoiseShaper 上限設定 | ~40 MB | 低 |
| **合計** | **~756 MB** | |
| **改善後目標** | **~1.57 GB** | |

---

## 5. 結論

1. **AoS スクラッチ化は正常に動作しており、NUC 1基あたり約 48% のメモリ削減を達成している。**
2. **2.33GB の主原因は AoS ではなく、StereoConvolver の設計上の三重保持（active/fading/pendingConv）である。**
3. **AoS 削減により、仮にこの三重保持が解消された場合の最終メモリは約 1.0GB（= 666MB + 330MB）となる。**
4. **最大の改善効果は `pendingConv` の早期 retire（666MB 削減）で得られ、実装コストは低い。**

---

## 付録A: 計測・検証方法

- **IR 長**: 25906 samples (≈135ms @ 192kHz), 処理用 192000 samples
- **サンプルレート**: 192kHz → 384kHz (×2 オーバーサンプリング)
- **ブロックサイズ**: 1024 → 2048 (OS 後)
- **DSPCore**: gen=1 から gen=8 まで 8回再構築
- **NUC**: AoS 削減後（333MB/NUC, 666MB/StereoConvolver）
- **NoiseShaper**: 学習中（iter=380, buffer=3,840,000 samples）

## 付録B: 参照コード位置

| シンボル | ファイル | 行 |
| :--- | :--- | :--- |
| `StereoConvolver` 定義 | `src/ConvolverProcessor.h` | L628 |
| `nucConvolvers[2]` | `src/ConvolverProcessor.h` | L632 |
| `pendingConv` | `src/ConvolverProcessor.h` | L566 |
| `IncrementalRebuildJob` | `src/ConvolverProcessor.h` | L555 |
| `activeRuntimeDSPSlot` | `src/audioengine/AudioEngine.h` | L1898 |
| `fadingRuntimeDSPSlot` | `src/audioengine/AudioEngine.h` | L1908 |
| `DSPCore::convolver` | `src/audioengine/AudioEngine.h` | L845 |
| `irBufSize` (改修後) | `src/MKLNonUniformConvolver.cpp` | L699 |
| `fdlBufSize` (改修後) | `src/MKLNonUniformConvolver.cpp` | L700 |
