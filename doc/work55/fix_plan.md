# コンボルバー改修計画書 v6.1

**バージョン**: 6.1
**作成日**: 2026-06-24
**最終確定日**: 2026-06-24

## 目次

1. [総合評価](#1-総合評価)
2. [F3: r8brain IRテイル切り捨て修正](#2-f3-r8brain-irテイル切り捨て修正)
3. [F5: applyAllpassToIR 削除](#3-f5-applyallpasstoir-削除)
4. [F2: mixedPreRingTau パラメータ削除](#4-f2-mixedpreringtau-パラメータ削除)
5. [F4: computeMasteringSizing 削除（追加調査後）](#5-f4-computemasteringsizing-削除追加調査後)
6. [F1: Mixed Phase クロスオーバー方向修正（DSP仕様変更・別リリース）](#6-f1-mixed-phase-クロスオーバー方向修正dsp仕様変更別リリース)
7. [F6: レイヤースケジューリング最適化（将来課題）](#7-f6-レイヤースケジューリング最適化将来課題)
8. [実施順序](#8-実施順序)

---

## 1. 総合評価

| ID | 項目 | 評価 | 優先度 | フェーズ |
| --- | --- | --- | --- | --- |
| F3 | r8brain出力長修正 | ★★★★★ 即実施推奨 | **高** | Phase 1 |
| F5 | 未使用コード削除 | ★★★★★ 即実施推奨 | 低 | Phase 2 |
| F2 | ダミーパラメータ削除 | ★★★★☆ 採用推奨 | 低 | Phase 3 |
| F1 | Mixed Phase方向修正 | ★★★☆☆ DSP仕様変更（設計者判断） | 低 | Phase 4（別リリース推奨） |
| F4 | 孤立計算削除 | ★★★☆☆ 保留案件 | 低 | Phase 5（最終） |
| F6 | レイヤースケジューリング | 妥当（将来課題） | 情報 | — |

---

## 2. F3: r8brain IRテイル切り捨て修正

### 2.1 現状

`src/IRDSP.cpp` の `IRDSP::resampleIR()` が出力バッファ長を以下の式で計算している：

```cpp
const double expectedLen = static_cast<double>(inLength) * ratio + 2.0;
const int maxOutLen = static_cast<int>(expectedLen);
```

### 2.2 問題

r8brain の `CDSPResampler::getLatency()` は常に `0` を返す（内部レイテンシ自動除去設計）。その代償として、入力終端後のフラッシュ処理で内部フィルタの残留サンプルを排出する必要がある。140dB/2% 設定では Harris の近似式 `N ≈ 140/(22·0.02) ≈ 318タップ` のフィルタ長となる。

現在の `+2.0` マージンでは：

1. バッファサイズが不足しフラッシュ出力を収容できない
2. 出力ループが `while (done < maxOutLen)` で打ち切られる
3. IR末尾（リバーブテイル等）が静かに消失する

### 2.3 修正方針

**ファイル**: `src/IRDSP.cpp`

**修正箇所A: バッファサイズ計算**

```cpp
// 変更前: +2.0 固定マージン
const double expectedLen = static_cast<double>(inLength) * ratio + 2.0;

// 変更後: getMaxOutLen() で適切な上限を取得
r8b::CDSPResampler tempResampler(inputSR, targetSR, inLength,
                                  cfg.transBand, cfg.stopBandAtten, cfg.phase);
const int maxOutLen = tempResampler.getMaxOutLen(inLength);
if (maxOutLen <= 0)
    return {};  // 不正な出力長の場合は空バッファを返す
```

**修正箇所B: チャンネル完了追跡とトリム**

現行コードはチャンネルごとに `done` をローカル変数として持つが、最後に `resampled.setSize(numCh, maxOutLen, ...)` で固定長を返している。各チャンネルでフラッシュ完了位置が異なる可能性があるため、最大値でトリムする必要がある。

```cpp
// 各スレッドの完了状態を管理
// futures.wait() が全スレッド完了の同期点となるため、
// vector<int> への異なるインデックス書き込みでデータ競合は発生しない。
std::vector<int> channelDone(numCh, -1);  // -1初期化: 例外・未完了チャンネルを識別可能
std::atomic<bool> anyChannelCancelled{false};

for (int ch = 0; ch < numCh; ++ch) {
    futures.emplace_back(std::async(std::launch::async, [&, ch]() {
        try {
            // キャンセルチェック: early-exit時は channelDone[ch]=-1 のまま
            if (shouldExit && shouldExit()) {
                anyChannelCancelled.store(true, std::memory_order_relaxed);
                return;
            }
            // ... existing processing ...
            // 各スレッドは自身の ch（ユニークなインデックス）のみ書き込む
            channelDone[ch] = done;
        } catch (...) {
            anyChannelCancelled.store(true, std::memory_order_relaxed);
            throw;  // get() で再送出
        }
    }));
}

for (auto& f : futures) f.get();  // get() を使用: 内部で例外が発生した場合も確実に伝播される（wait() では例外が回収されない）

if (anyChannelCancelled.load(std::memory_order_relaxed))
    return {};  // キャンセル時は空バッファを返す

// 全チャンネル中の最大完了位置でトリム
// maxDone 採用理由: 理論上は全チャンネル同一長となるが、安全のため maxDone を採用。
// minDone を使うと最長チャンネルの末尾が切り捨てられる可能性がある。
// ただし getMaxOutLen() が理論上の最大出力長であり、done==maxOutLen のケースもあるため、
// maxDone < maxOutLen の時のみ縮小する条件付きトリムとする。
const int maxDone = *std::max_element(channelDone.begin(), channelDone.end());
if (maxDone < 0)
    return {};  // 全チャンネル未完了（例外発生等）の場合は空バッファ
if (maxDone < maxOutLen)
    resampled.setSize(numCh, maxDone, true, true, true);
// maxDone == maxOutLen の場合は元のバッファサイズを維持（現状コードとの互換性）

// 注意: IRDSP::resampleIR の戻り値型は juce::AudioBuffer<double> であり、
// 呼び出し側（IRConverter.cpp）は空バッファをエラーとして扱う。
// 戻り値型が ResampleOutput 構造体の場合は Cancelled ステータスを返すこと。
```

### 2.4 API確認

`r8b::CDSPResampler::getMaxOutLen()` は `r8brain-free-src/CDSPResampler.h` L502 に `virtual int getMaxOutLen(const int) const` として存在を確認済み。r8brain の公開APIの一部であり、本バージョンでも利用可能。

### 2.5 影響確認

| 項目 | 結果 |
| --- | --- |
| 影響パス | `IRConverter.cpp`（`IRDSP::resampleIR()`）のみ |
| メインローダーパス | 安全。`getMaxOutLen()` + `oneshot()` 使用済み |
| 性能 | `getMaxOutLen()` は O(1) |
| 既存キャッシュ | 出力IR長が変わるため PreparedIRState キャッシュは無効化（期待動作） |

---

## 3. F5: `applyAllpassToIR` 削除

### 3.1 現状

`AllpassDesigner.h` L115 で宣言・`AllpassDesigner.cpp` L595-742 で実装されているが、**呼び出し箇所が一つもない**。

### 3.2 最終確認結果

```
$ grep -r "AllpassDesigner::applyAllpassToIR" src/
  → src/AllpassDesigner.cpp:595  (定義のみ)

$ grep -r "applyAllpassToIR(" src/
  → src/AllpassDesigner.cpp:595  (定義)
  → src/AllpassDesigner.h:115    (宣言)

$ grep -r "applyAllpassToIR" doc/
  → doc/class_definition_en.md, doc/class_definition_jp.md  (APIドキュメント)
  → doc/work/bug4_observation_register.md  (既存認識: 呼び出し0件)
  → doc/plan4.md  (デッドコード認識あり)
```

呼び出し元は存在しない。削除推奨。

### 3.3 削除手順

1. `AllpassDesigner.h` L115 の宣言を削除
2. `AllpassDesigner.cpp` L588-742 の実装（コメント含む）を削除
3. `doc/class_definition_en.md` / `doc/class_definition_jp.md` の該当行を削除
4. ビルド確認

---

## 4. F2: `mixedPreRingTau` パラメータ削除

### 4.1 現状

`mixedPreRingTau` は：

- `convertToMixedPhaseAllpass()` 内でキャッシュキーにのみ使用
- `AllpassDesignerConfig` に `tau` フィールドなし
- `convertToMixedPhaseFallback()` で `(void)tau`
- 音響的出力に全く影響しないダミーパラメータ

### 4.2 削除範囲（17ファイル）

| # | ファイル | 削除内容 |
| --- | --- | --- |
| 1 | `ConvolverProcessor.h` | `MIXED_TAU_MIN/MAX/DEFAULT` 定数 |
| 2 | `ConvolverProcessor.h` | `BuildSnapshot::mixedPreRingTau` |
| 3 | `ConvolverProcessor.h` | `IRCacheKey::tau` + `operator<` 該当行 |
| 4 | `ConvolverProcessor.h` | `setMixedPreRingTau()` / `getMixedPreRingTau()` 宣言 |
| 5 | `convolver/ConvolverProcessor.Runtime.cpp` | setter/getter 実装 |
| 6 | `convolver/ConvolverProcessor.StateAndUI.cpp` | hash/serialize/deserialize の全参照 |
| 7 | `convolver/ConvolverProcessor.LoaderThreadInline.h` | BuildSnapshot メンバ |
| 8 | `convolver/ConvolverProcessor.LoaderThread.cpp` | コンストラクタ引数 + メンバ初期化子 |
| 9 | `convolver/ConvolverProcessor.LoadPipeline.cpp` | `buildSnapshot.mixedPreRingTau` 2箇所 |
| 10 | `convolver/ConvolverProcessor.MixedPhase.cpp` | `key.tau` 5箇所 + `(void)tau` |
| 11 | `ConvolverControlPanel.h` | `mixedTauSlider`/`mixedTauLabel` + `pendingMixedTau*` |
| 12 | `ConvolverControlPanel.cpp` | 全 `mixedTau*` 参照（〜20行） |
| 13 | `audioengine/AudioEngine.h` | `setConvolverMixedPreRingTau()` 宣言 |
| 14 | `audioengine/AudioEngine.Parameters.cpp` | setter実装 + `"mixedTau"` バリデーション |
| 15 | `MainWindow.cpp` | `--cli-pre-ring-tau` CLIパラメータ処理（L670-680） |
| 16 | `MixedPhasePersistentCache.cpp` | 全 `tau` 引数（5関数） |
| 17 | `MixedPhasePersistentCache.h` | 同上の宣言 |

### 4.3 DiskHeader 互換性と kLastUsedTimeOffset 修正

`MixedPhasePersistentCache::DiskHeader`（`MixedPhasePersistentCache.h` L65-82）に `float tau;` フィールドが含まれている。`tau` 削除時にはこのフィールドも削除し、同時に `kVersion` を `1` から `2` に更新する必要がある。

```cpp
// 現行:
static constexpr uint32_t kVersion = 1;
struct DiskHeader {
    ...
    float tau;            // ← 削除
    ...
};

// 修正後:
static constexpr uint32_t kVersion = 2;  // ← バージョンアップ
struct DiskHeader {
    ...
    // float tau; 削除
    ...
};
```

`kVersion` を更新しない場合、旧フォーマットのキャッシュファイルを別の構造体サイズで読み込もうとし、デシリアライズエラーまたはガベージデータの読み取りが発生する。`header.version != kVersion` のチェック（`MixedPhasePersistentCache.cpp` L98）により自動的に無効化され新規作成されるため安全だが、明示的なバージョン管理が必要。

**`kLastUsedTimeOffset` の修正（必須）**: `MixedPhasePersistentCache::touch()` 内（`MixedPhasePersistentCache.cpp` L256）にハードコードされたバイトオフセット定数がある：

```cpp
// 現行: tau の存在を前提とした固定値
static constexpr int kLastUsedTimeOffset = 52;
```

これは `DiskHeader` 内の `lastUsedTime` フィールドの位置を指す。`tau`（4byte、offset 44）を削除すると `lastUsedTime` のオフセットは 52 → 48 に変化するため、固定値は正しく動作しなくなる。

**修正方法**: 固定値を廃止し、コンパイル時オフセット計算に変更する：

```cpp
// 修正後: 構造体レイアウトに依存しない安全な方法
#include <cstddef>  // offsetof
const size_t lastUsedTimeOffset = offsetof(DiskHeader, lastUsedTime);
```

`offsetof` はコンパイル時に正しいオフセットを計算するため、`DiskHeader` のフィールド追加・削除に対して自動追従する。さらに、将来の構造体変更時に予期せぬオフセット変化を検出するため、以下の `static_assert` を `DiskHeader` 定義直後に追加することを推奨：

```cpp
// DiskHeader レイアウト不変契約
// サイズを明示的にアサート（現状の128バイト制約を維持）
static_assert(sizeof(DiskHeader) <= 128, "DiskHeader must not exceed 128 bytes");
// lastUsedTime が構造体内に収まることを確認（固定値アサートは保守性低下を招くため非推奨）
static_assert(offsetof(DiskHeader, lastUsedTime) + sizeof(uint64_t) <= sizeof(DiskHeader),
              "lastUsedTime must fit within DiskHeader");
```

`#pragma pack(push, 1)` が適用されたパック構造体であるため、コンパイラ依存のパディングが入らないことを前提とした明示的なサイズ保証が有効。

### 4.4 キャッシュ影響

`IRCacheKey` からの `tau` 削除＋`DiskHeader` からの `tau` 削除＋`kVersion 1→2` の変更により、メモリ・ディスク両方のキャッシュが完全に無効化される。次回起動時に Mixed Phase IR の再設計が一度発生するが許容範囲。

**F1 との関係**: メモリキャッシュ（`std::map<IRCacheKey, CacheEntry> irCache`）は `IRCacheKey` にクロスオーバー方向情報を含まない。そのため F1（方向変更）を単独実施した場合、アプリケーション起動中のメモリキャッシュでも旧方向の Mixed IR がヒットする。F1 のキャッシュ無効化は F2 の `kVersion` 更新（＋`DiskHeader` レイアウト変更）に依存しており、F1 と F2 は同時リリースする必要がある。

**メモリキャッシュ無効化の推奨設計**: `kVersion` は永続キャッシュのみを制御するため、メモリキャッシュには別の仕組みが必要。推奨は `IRCacheKey` に `uint8_t algorithmVersion` フィールドを追加すること：

```cpp
struct IRCacheKey {
    uint64_t fileHash;
    double sampleRate;
    PhaseMode phaseMode;
    float f1, f2;
    int targetLength;
    uint8_t algorithmVersion = 0;  // 追加: Mixed Phaseアルゴリズム版。変更時にインクリメント
};
```

これによりメモリキャッシュのキーが変わるため、方向変更時に自動的にミスヒットする。`kVersion`（永続キャッシュ）と `algorithmVersion`（メモリキャッシュ）の併用が最も安全。

### 4.5 実装前の最終確認

```bash
# tau が DSP 処理に使われていないことの最終確認
grep -rn "\.tau" src/convolver/ConvolverProcessor.MixedPhase.cpp
# → キャッシュキー代入（key.tau = ...）のみであることを確認

# 削除対象ファイルの網羅性確認
grep -rn "mixedPreRingTau\|MIXED_TAU\|mixedTauSlider\|setMixedPreRingTau\|getMixedPreRingTau" src/ --include="*.cpp" --include="*.h"
# → 計画書記載の17ファイルで過不足ないことを確認
```

v5.2 時点の確認結果：全 `.tau` 参照はキャッシュキー代入（`key.tau = ...`）のみ。DSP処理（群遅延計算、オールパス設計、周波数応答合成）への影響はゼロ。

### 4.6 PersistentCache シグネチャ変更チェックリスト

`MixedPhasePersistentCache` の以下の全関数で `tau` 引数を削除する必要がある。実装時の見落とし防止のためリスト化：

| 関数 | ファイル | 変更内容 |
| --- | --- | --- |
| `computeKeyHash(... float tau ...)` | `MixedPhasePersistentCache.cpp` L19 | `tau` 引数削除 + `uint32_t tauBits` 削除 + `hashCombine(h, tauBits)` 削除 |
| `getCacheFile(... float tau ...)` | `MixedPhasePersistentCache.cpp` L58 | `tau` 引数削除 |
| `load(... float tau ...)` | `MixedPhasePersistentCache.cpp` L74 | `tau` 引数削除 + 呼び出し側修正 |
| `save(... float tau ...)` | `MixedPhasePersistentCache.cpp` L149 | `tau` 引数削除 + `header.tau = tau;` 削除 |
| `touch(... float tau ...)` | `MixedPhasePersistentCache.cpp` L227 | `tau` 引数削除 |
| `remove(... float tau ...)` | `MixedPhasePersistentCache.cpp` L312 | `tau` 引数削除 |
| `MixedPhasePersistentCache.h` 全宣言 | L17-L48 | 同上の宣言から `tau` 引数を削除 |

---

## 5. F4: `computeMasteringSizing` 削除（追加調査後）

### 5.1 現状と調査結果

`ConvolverProcessorInternal::computeMasteringSizing()`（`ConvolverProcessor.Internal.h` L116-137）は `firstPartition`/`maxFFTSize` を計算する。

**データフロー完全追跡結果**:

```
computeMasteringSizing(internalBlockSize, irLength)
  → init(..., maxFFTSize, ..., firstPartition, ...)       [Lifecycle.cpp / LoaderThread.cpp]
  → storedMaxFFTSize = maxFFTSize                          [ConvolverProcessor.h L717]
  → storedFirstPartition = firstPartition                  [ConvolverProcessor.h L719]
  → SetImpulse(irData, irLen, knownBlockSize, scale,       [ConvolverProcessor.h L733]
               enableDirectHead, filterSpec)
    ※ maxFFTSize / firstPartition は引数に存在せず、NUCに届かない
  → clone() → init(storedMaxFFTSize, ..., storedFirstPartition)  [ConvolverProcessor.h L776]
    ※ 再び init() に戻るだけの循環。SetImpulse には依然として届かない
```

**`storedMaxFFTSize` / `storedFirstPartition` の全参照箇所**（6箇所のみ）：

| 行 | ファイル | 内容 |
| --- | --- | --- |
| L648 | `ConvolverProcessor.h` | `int storedMaxFFTSize = 0;`（メンバ宣言） |
| L650 | `ConvolverProcessor.h` | `int storedFirstPartition = 0;`（メンバ宣言） |
| L717 | `ConvolverProcessor.h` | `storedMaxFFTSize = maxFFTSize;`（代入） |
| L719 | `ConvolverProcessor.h` | `storedFirstPartition = firstPartition;`（代入） |
| L776 | `ConvolverProcessor.h` | `clone()` → `init(storedMaxFFTSize, ..., storedFirstPartition)`（再投入） |

**診断**: 上記以外の参照（ログ出力、デバッグ表示、互換レイヤ、将来拡張のフック）は**一切存在しない**。

**追加調査結果**: `clone()` は `shareConvolutionEngineFrom()`（`StateAndUI.cpp` L436）からのみ呼ばれる。しかし `shareConvolutionEngineFrom()` 自体はコードベース内で**呼び出し元が存在しない**（grep 0件）。`ConvolverProcessor` の公開APIとして宣言されているが、現在は休眠状態である。したがって `storedMaxFFTSize`/`storedFirstPartition` は宣言・代入・休眠clone経路での循環のみで完結しており、NUC構成に影響する経路は完全に存在しない。

### 5.2 判断

「死んでいる可能性が高い」が「完全削除を断定できる証拠がまだ十分とは言えない」状態。

**最終確認用コマンド（Phase 4 直前に実行）**:

```bash
grep -rn storedMaxFFTSize src/
grep -rn storedFirstPartition src/
```

v4.1 時点の確認結果：`storedMaxFFTSize` は3行・`storedFirstPartition` は3行、いずれも `ConvolverProcessor.h` 内の宣言・代入・clone の3箇所のみ。0件（完全に未使用）ではないが、NUC 構成に影響しない「孤立した保持」であることが確認済み。

### 5.3 推奨方針

**Phase 4（最後）に回し、Phase 4 直前に再度 `rg storedMaxFFTSize` / `rg storedFirstPartition` を実行し、参照箇所が増えていないことを確認した上で、以下の判断基準を満たした場合のみ削除を実施する**：

| 判断基準 | 確認方法 |
| --- | --- |

| 判断基準 | 確認方法 |
| --- | --- |
| 全ビルド構成で `computeMasteringSizing` 削除後にコンパイル成功 | Release/Debug 両方でビルド |
| `init()` 引数削除後も `clone()` が正常動作 | ユニットテストまたはIRロード→再構成の動作確認 |
| NUCのパーティションサイズが削除前と同一 | `getLatency()` の戻り値を比較 |
| IRロード→ホットスワップ→IR再ロードのサイクルが正常 | 手動テスト |

### 5.4 暫定措置（Phase 4までの間）

現状のコードは動作に支障がないため、緊急の対応は不要。ただし、将来の保守者への誤解を防ぐため、以下のコメントを `computeMasteringSizing()` に追記することを推奨：

```cpp
// NOTE: [OBSOLETE] この関数の計算結果は NUC 構成に反映されない。
// storedMaxFFTSize / storedFirstPartition として保持されるが、
// MKLNonUniformConvolver::SetImpulse() はこれらの引数を受け取らず、
// パーティションサイズを blockSize から独自計算する。
// 削除判断の詳細は doc/work55/fix_plan.md §5 を参照。
```

---

## 6. F1: Mixed Phase クロスオーバー方向修正（DSP仕様変更・別リリース）

### 6.1 位置づけ

**設計者の最終判断：DSP仕様変更として、クロスオーバー方向を反転する。現状の「低域=Linear／高域=Minimum」はコード上の誤りではなく、設計者の意図した「低域=Minimum／高域=Linear」と異なる設計思想に基づく実装であるため、設計判断として修正する。**

> 設計者意見：「F1は設計思想から実装をしてください。」
> 設計者追補：「現状は誤実装なので、F1を実施してください。」

**注意**: 「誤実装」とはコード自体の論理誤りではなく、設計者の意図した Mixed Phase の方向と実装が逆であることを指す。DSP 理論上は現行方向（低域Linear/高域Minimum）にもルーム補正用途としての合理性があり、「一般的な EQ 用途と比較して誤り」という意味ではない。

### 6.2 修正内容

`convertToMixedPhaseAllpass()`（L303-311）および `convertToMixedPhaseFallback()`（L810-818）の重み関数 `wLinear`/`wMinimum` の相互に `1-w` 関係を反転する。

**現行（設計思想に対して誤実装）**:

```cpp
// 低域=Linear Phase, 高域=Minimum Phase
// 設計者の意図と逆
```

修正後:

```cpp
// 低域=Minimum Phase（プリリンギング抑制）, 高域=Linear Phase（位相情報保存）
// 設計者の意図に合致
```

**ファイル**: `src/convolver/ConvolverProcessor.MixedPhase.cpp`（Allpass版 + Fallback版の2箇所）

### 6.3 変更コード

**修正A: `convertToMixedPhaseAllpass()` 内**

```cpp
// 変更前: 低域=Linear Phase, 高域=Minimum Phase
double wLinear = 1.0;
if (freq >= transitionHiHz)
    wLinear = 0.0;
else if (freq > transitionLoHz) {
    const double x = (freq - transitionLoHz) * invSpan;
    wLinear = 0.5 * (1.0 + std::cos(juce::MathConstants<double>::pi * x));
}
const double wMinimum = 1.0 - wLinear;

// 変更後: 低域=Minimum Phase, 高域=Linear Phase
double wMinimum = 1.0;
if (freq >= transitionHiHz)
    wMinimum = 0.0;
else if (freq > transitionLoHz) {
    const double x = (freq - transitionLoHz) * invSpan;
    wMinimum = 0.5 * (1.0 + std::cos(juce::MathConstants<double>::pi * x));
}
const double wLinear = 1.0 - wMinimum;
```

**修正B: `convertToMixedPhaseFallback()` 内**: 同様。

### 6.4 キャッシュ整合性問題（重要）

**警告**: 現行の `MixedPhasePersistentCache::DiskHeader` はキャッシュキーにクロスオーバー方向情報を含まない。そのため、F1 を実施しても旧方向の Mixed IR がキャッシュから再利用され、修正が反映されない。

**解決策（必須）**: 以下のいずれかを実施すること。

| 案 | 方法 | 影響 |
| --- | --- | --- |
| **案A（推奨）** | `MixedPhasePersistentCache::kVersion` を `1` → `2` に更新（F2と同様） | 全 Mixed Phase キャッシュが無効化→再設計。最も安全 |
| **案B** | `IRCacheKey` に `uint8_t mixDirection` を追加＋`DiskHeader` に対応フィールド追加 | キャッシュキー構造変化で旧キャッシュ自動無効。差分が明確 |

案A が F2 と同じ機構で実装できるため推奨。ただし、以下のリリースパターンに応じて管理方法が異なる：

| リリースパターン | kVersion管理 | 備考 |
| --- | --- | --- |
| **F2とF1を同時リリース** | F2の `kVersion 1→2` で両方の変更によるキャッシュ無効化が一度に完了。F1側で別途 `kVersion` 更新不要 | 最も効率的 |
| **F2を先にリリース、F1を後にリリース** | F1単独で `kVersion 2→3`（またはF2未適用なら `1→2`）を更新する必要あり。`DiskHeader` のフォーマット変更がない場合でも、方向変更を既存キャッシュに認識させるために必須 | 管理上明確 |

**推奨手順**: F1とF2を同じリリースに含める場合、F2の `kVersion=1→2` のみで十分。F1がF2より後のリリースになる場合は、F1専用で `kVersion` をインクリメントすること。

### 6.5 影響

| 項目 | 影響 |
| --- | --- |
| 音響的挙動 | **大**。IRの位相特性が反転。低域Minimum Phaseによりプリリンギング抑制、高域Linear Phaseにより位相情報保存 |
| キャッシュ | **要対応**（§6.4 参照）。F2 と同時実施で自動解決 |
| UIパラメータ | 変更なし（`MIXED_F1_DEFAULT_HZ`/`MIXED_F2_DEFAULT_HZ` はそのまま） |
| Mixed Phase IR品質 | 設計者の意図通りに改善される見込み |

### 6.6 実施条件

F1 の実施判断には以下を推奨（必須ではない）：

1. 変更前後の **Excess Group Delay 比較**（20-500Hz）
2. 変更前後の **Step Response 比較**
3. 変更前後の **Impulse Response 比較**（プリリンギング量）
4. **Null Test**（両IRの差信号を分析）
5. **Magnitude Response 比較**（振幅特性に差が出ないことを確認）
6. **Excess Phase 比較**（Mixed Phase変更の本質はExcess Phase特性の変化であるため直接確認が望ましい）

### 6.6 リリースノート注意

F1 は DSP 仕様変更であり、バグ修正ではない。ユーザー視点では「同じ設定（F1/F2/tau同じ）なのに音が変わった」と認識される。そのため F1 を含むリリースには以下の明記を推奨：

> **Mixed Phase アルゴリズム変更**: クロスオーバー方向を「低域=Minimum Phase／高域=Linear Phase」に修正しました。これにより Mixed Phase モード使用時の IR 位相特性が従来と変わります。既存の Mixed Phase キャッシュは自動的に無効化され、次回 IR ロード時に再設計が行われます。

### 6.7 実施順序上の注意

F1 は F3（`IRDSP.cpp`）とファイルが異なる。ただしF1はDSP仕様変更のため別リリース推奨。現行の推奨順序は **F3 → F5 → F2（リリース可能）→ F1（別リリース）→ F4**。

---

## 7. F6: レイヤースケジューリング最適化（将来課題）

### 7.1 現状

`MKLNonUniformConvolver::SetImpulse()` 内：

```cpp
const int l0Part = juce::nextPowerOfTwo(std::max(blockSize, 64));
const int l1Part = l0Part * tailL1L2Mult;   // デフォルト8
const int l2Part = l1Part * tailL1L2Mult;   // デフォルト8
```

Gardner (1995) 固定比率ヒューリスティック。実運用上問題なし。

### 7.2 対応

優先度最低。本改修フェーズでは実施しない。

---

## 8. 実施順序

```
Phase 1: F3 r8brain出力長修正（Bug Fix・最重要）
  → IRDSP.cpp
  ※ getMaxOutLen() APIはr8brain CDSPResampler.h L502で確認済み

Phase 2: F5 未使用コード削除（コード整理）
  独立実施可能・副作用最小
  → AllpassDesigner.h/cpp

Phase 3: F2 ダミーパラメータ削除（コード整理）
  → 17ファイル（変更量大・注意）
  ※ kVersion: 1→2 更新、kLastUsedTimeOffset→offsetof 変更、sizeof/offsetof static_assert 追加を含む
  ※ CLI --cli-pre-ring-tau も削除対象（MainWindow.cpp）
---- この時点でリリース可能（F3/F5/F2はバグ修正＋コード整理） ----

Phase 4: F1 Mixed Phaseクロスオーバー方向修正（DSP仕様変更・別リリース推奨）
  → ConvolverProcessor.MixedPhase.cpp（2箇所）
  ※ 「バグ修正」ではなくDSP仕様変更。音が変わるため別リリースが安全
  ※ キャッシュ整合性: F2のkVersion変更により自動解決（§6.4参照）
  ※ 実施前のExcess Group Delay/Step Response/Impulse Response比較を推奨
  ※ ユーザー価値のある仕様変更であり、単なるコード整理（F4）より優先

Phase 5: F4 孤立計算削除（追加調査後・保留案件）
  十分なコード追跡と検証を経てから判断
  → 5ファイル（変更量小）
  ※ 削除確定ではなく「保留案件」。OBSOLETEコメント追加も可

F6: 将来課題（本フェーズでは実施しない）
```

### 8.1 各フェーズの成果物

| Phase | 成果物 |
| --- | --- |
| 1 | 修正済み `IRDSP.cpp` + F3テイル確認ログ |
| 2 | 修正済み `AllpassDesigner.h/cpp` + ビルド成功確認 |
| 3 | 修正済み17ファイル（kVersion: 1→2、kLastUsedTimeOffset→offsetof、CLI削除含む）+ ビルド成功確認 |
| 4 | 修正済み `ConvolverProcessor.MixedPhase.cpp` + 位相特性確認ログ |
| 5 | 削除済み5ファイル + 動作確認ログ。または `computeMasteringSizing()` へのコメント追記 |
| 2 | 修正済み `AllpassDesigner.h/cpp` + ビルド成功確認 |
| 3 | 修正済み16ファイル + ビルド成功確認 |
| 4 | 削除済み5ファイル + 動作確認ログ。または `computeMasteringSizing()` へのコメント追記 |

---

## 付録A: コード行番号リファレンス

| ファイル | 行 | 内容 |
| --- | --- | --- |
| `src/IRDSP.cpp` | 18 | `+2.0` マージン（F3要修正） |
| `src/IRDSP.cpp` | 23 | `maxOutLen` バッファ確保（F3要修正） |
| `src/IRDSP.cpp` | 49-52 | フラッシュループ + `done < maxOutLen`（F3要修正） |
| `src/IRDSP.cpp` | 103 | `resampled.setSize(numCh, maxOutLen)`（F3要修正: maxDoneでトリム） |
| `src/convolver/ConvolverProcessor.MixedPhase.cpp` | 303-311 | Allpass版 クロスオーバー重み（F1要修正） |
| `src/convolver/ConvolverProcessor.MixedPhase.cpp` | 734 | `(void)tau;`（F2削除対象） |
| `src/convolver/ConvolverProcessor.MixedPhase.cpp` | 810-818 | Fallback版 クロスオーバー重み（F1要修正） |
| `src/convolver/ConvolverProcessor.Internal.h` | 116-137 | `computeMasteringSizing()`（F4削除候補） |
| `src/ConvolverProcessor.h` | 700-735 | `StereoConvolver::init()`（F4引数整理候補） |
| `src/convolver/ConvolverProcessor.Lifecycle.cpp` | 217-221 | `computeMasteringSizing` 呼び出し（F4削除候補） |
| `src/convolver/ConvolverProcessor.LoaderThread.cpp` | 219 | `computeMasteringSizing` 呼び出し（F4削除候補） |
| `src/AllpassDesigner.h` | 115 | `applyAllpassToIR` 宣言（F5削除候補） |
| `src/AllpassDesigner.cpp` | 588-742 | `applyAllpassToIR` 実装（F5削除候補） |

---

*本計画書は 2026-06-24 時点の `lonewolf-jp/ConvoPeq` main ブランチのソースコードに基づく。v6.1 での確定: F3のchannelDone初期値を -1 に変更しmaxDone<0判定で例外・未完了を識別。v4.0から21回の反復を経て全項目確定。*
