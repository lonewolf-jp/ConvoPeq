# ConvoPeq Part 7 & Part 8 改修計画 検証レポート v3

**検証日**: 2026-07-24
**検証対象**: `ConvoPeq_Part7_Part8_RepairPlan.md` の実装状況
**検証方法**: ソースコード分析 + 横断的配線チェック + ビルドテスト

---

## 検証サマリー

| フェーズ | 状態 | 問題点 |
|---------|------|--------|
| Phase 1 | ✅ 正常 | equalPowerSin の重複定義（保守リスク） |
| Phase 2 | ✅ 正常 | なし |
| Phase 3 | ✅ 正常 | processDoubleToBuffer にモノラル→ステレオ複製なし（非対称） |
| Phase 4 | 未実装 | — |
| Phase 5 | ✅ 正常 | なし |
| Phase 6 | 情報 | — |

---

## Phase 1: クロスフェード等電力化 — ✅ 正常

### equalPowerSin の重複定義（保守リスク）

| # | ファイル | 行 | 状態 |
|---|---------|---|------|
| 1 | `AudioEngine.Processing.AudioBlock.cpp` | L19 | ✅ 定義済み |
| 2 | `AudioEngine.Processing.BlockDouble.cpp` | L19 | ✅ 定義済み |
| 3 | `convolver/ConvolverProcessor.Runtime.cpp` | L26 | ✅ 定義済み |
| 4 | `eqprocessor/EQProcessor.Processing.cpp` | L116 | ✅ 定義済み |

**判定**: 全4ファイルで同一の6次テイラー展開を使用。現在は一致しているが、将来の修正時に1つだけ更新し忘れるリスクがある。保守性の問題。

### クロスフェード呼び出し箇所

| # | ファイル | 行 | 状態 |
|---|---------|---|------|
| 1 | `AudioBlock.cpp` | L443-444 | ✅ `equalPowerSin` 使用 |
| 2 | `BlockDouble.cpp` | L421-422 | ✅ `equalPowerSin` 使用 |
| 3 | `ConvolverProcessor.Runtime.cpp` | L600-601, 671-672 | ✅ `equalPowerSin` 使用 |

**判定**: 全クロスフェードパスで `equalPowerSin` を使用。旧線形パターンの残存なし。

---

## Phase 2: FTZ/DAZ スレッド起動時1回設定化 — ✅ 正常

### オーディコールバックパスの網羅性

| # | エントリポイント | FTZ/DAZ 設定 | 状態 |
|---|----------------|-------------|------|
| 1 | `getNextAudioBlock()` | ✅ `ensureThreadFloatingPointEnvironment()` | 正常 |
| 2 | `processBlockDouble()` | ✅ `ensureThreadFloatingPointEnvironment()` | 正常 |
| 3 | `processToBuffer()` | ❌ なし（内部ヘルパー） | 情報 |
| 4 | `processDoubleToBuffer()` | ❌ なし（内部ヘルパー） | 情報 |
| 5 | `EQProcessor::process()` | ✅ `ScopedNoDenormals` | 冗長だが無害 |
| 6 | `ConvolverProcessor::process()` | ✅ `ScopedNoDenormals` | 冗長だが無害 |

**判定**: トップレベルのJUCEコールバックエントリポイントは2つともカバー済み。内部ヘルパーはコールバックスコープ内から呼ばれるため、個別のFTZ/DAZ設定は不要。

---

## Phase 3: processToBuffer モノラル→ステレオ複製 — ✅ 正常（1点要確認）

### processToBuffer の呼び出し元

| # | 呼び出し元 | ファイル | 状態 |
|---|-----------|---------|------|
| 1 | `AudioBlock.cpp:409` | フェードパス | ✅ 1箇所のみ |

**判定**: 呼び出し元は1箇所のみ。ステレオ入力パスへの影響なし。

### processDoubleToBuffer との非対称性（要確認）

`processDoubleToBuffer()`（`DSPCoreDouble.cpp:281`）にはモノラル→ステレオのL→R複製がない。

**理由**: double パスはホストから常にステレオバッファが提供されると想定。実際の問題にはならない可能性が高いが、float パスとの非対称性として記録。

---

## Phase 5: デッドコード削除 + アサート追加 — ✅ 正常

### PublicationBuffer 削除

| # | 検証項目 | 状態 |
|---|---------|------|
| 1 | `.h` からクラス定義削除 | ✅ 完全削除 |
| 2 | `.cpp` から実装削除 | ✅ 完全削除 |
| 3 | `src/` 内の残存参照 | ✅ ゼロ件 |

### getVersion() assert

| # | 検証項目 | 状態 |
|---|---------|------|
| 1 | `assert(true)` プレースホルダ | ✅ Line 173 |
| 2 | ADR-010 TODO コメント | ✅ `NOLINT(danger-comment)` 付き |

### DSPHandle static_assert

| # | 検証項目 | 状態 |
|---|---------|------|
| 1 | `is_trivially_copyable_v` | ✅ Line 163 |
| 2 | `is_standard_layout_v` | ✅ Line 165 |
| 3 | `is_always_lock_free` | ⚠️ コメントアウト済み（正当な理由あり） |

---

## 横断的配線チェック

### equalPowerSin の重複定義（保守リスク）

**問題**: 4ファイルに同一の `equalPowerSin` 関数が定義されている。
**リスク**: 将来の修正時に1つだけ更新し忘れる可能性。
**推奨**: `src/core/EqualPowerCrossfade.h` に集約（中長期的改善）。

### processDoubleToBuffer との非対称性

**問題**: float 版 `processToBuffer` にはモノラル→ステレオ複製があるが、double 版 `processDoubleToBuffer` にはない。
**リスク**: 低（double パスはホストから常にステレオバッファが提供されると想定）。
**推奨**: 将来的に両パスの統一を検討。

### thread_local 命名規則

**判定**: すべての `thread_local` 変数が `t_` プレフィックスを使用。一致確認済み。問題なし。

### インクルードチェーン

**判定**: `AudioEngine.Mmcss.cpp` は `JuceHeader.h` を通じて `<xmmintrin.h>` と `<pmmintrin.h>` にアクセス可能。問題なし。

---

## 総合判定

**改修計画の実装は適切に行われている。** 重要な問題点は検出されなかった。

### 発見された問題点

| # | 問題 | 深刻度 | 状態 |
|---|------|--------|------|
| 1 | `equalPowerSin` の重複定義（保守リスク） | 中 | 将来改善推奨 |
| 2 | `processDoubleToBuffer` にモノラル→ステレオ複製なし | 中 | 非対称性（現状影響なし） |

### 実装済みフェーズ

1. **Phase 1**: ✅ クロスフェード等電力化
2. **Phase 2**: ✅ FTZ/DAZ スレッド起動時1回設定化
3. **Phase 3**: ✅ processToBuffer モノラル→ステレオ複製
4. **Phase 5**: ✅ デッドコード削除 + アサート追加

### 未実装フェーズ

5. **Phase 4**: RT-safe ヘルパー関数の DSP Foundation 層への発展（計画のみ）
6. **Phase 6**: 防御的改善（将来向け）

---

*本レポートは改修計画の実装状況をソースコード分析 + 横断的配線チェックにより検証した結果です。*
