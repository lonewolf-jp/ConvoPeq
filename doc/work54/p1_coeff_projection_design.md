# P1: RuntimeBuilder coefficient projection 修正 — 設計書

> 作成日: 2026-06-24
> 関連: Phase C 本採用マージ完了 (commit 9286306)
> 位置づけ: Practical Stable ISR Bridge Runtime 完成度向上

## 1. 概要

### 問題

`RuntimeBuilder::buildRuntimePublishWorld()` が係数情報を常に 0 に設定している:

```cpp
// src/audioengine/RuntimeBuilder.cpp 付近 L345
worldOwner->coefficient.adaptiveCoeffBankIndex = 0;  // 実際は 107 等
worldOwner->coefficient.adaptiveCoeffGeneration = 0;  // 実際は 11 等
worldOwner->coefficient.eqCoeffHash = 0;
```

### 影響

- RuntimeWorld の coefficient semantic が実態を投影できていない
- `semanticHash.coefficientHash` が実質定数（係数変更を検出しない）
- `[COEFF_AUTH] worldGen=0` と表示され診断価値が低い
- テスト `RuntimeWorldAuthorityProjectionTests.cpp` が期待するパターンと乖離

### 非目標

- **RuntimeWorld.generation == bank.generation は目標にしない**
  AdaptiveCoeffBank は独立 Authority のため、Learner 稼働中は
  `RuntimeWorld.generation < live bank.generation` が常時発生し得る（正常）。

## 2. Authority モデル（確定）

```
[Runtime Structural Authority]          [Adaptive Learning Authority]
  RuntimeWorld                            AdaptiveCoeffBank
  ├→ routing / automation / execution    ├→ coeffSetA/B + activeIndex
  │   = Publish 経由                     │   = atomic 直接更新
  ├→ topology / overlap                  ├→ generation（atomic increment）
  │   = Publish 経由                     ├→ state（mutex 保護）
  └→ coefficient （Projection only）      └→ autosave（bank 直接読取）
       ↑ Publish開始時点の bank 値を投影
       ↑ 現状は常に0（本P1で修正）
```

### Authority 間の契約

1. `AdaptiveCoeffBank` = Adaptive Learning の **Source of Truth**
2. `RuntimeWorld.coefficient` = Publish **開始時点**の bank 値の **投影（Projection）**
3. Learner 稼働中は `RuntimeWorld.coefficientGeneration < live bank.generation` が常時発生し得る（**正常動作**）
4. `worldGen <= liveGen` は**契約としない**（bank切替により一瞬 `worldGen > liveGen` が観測され得る）。
   確認すべきは `worldGen > liveGen` が継続的に観測されないこと。
5. `payloadHash` は係数変更を検出しない（Structural 変更ではないため設計として正しい）

## 3. 修正内容

### 3.1 RuntimeBuilder.cpp の修正

**ファイル**: `src/audioengine/RuntimeBuilder.cpp`
**関数**: `buildRuntimePublishWorld()`
**場所**: 既存の atomic 読取ブロック内（L255〜L290 のパラメータ収集ブロックに追加）

**追加するコード**:

```cpp
// Coefficient fields: Publish開始時点の live 値を投影
// ★ ISR Runtime 契約: RuntimeWorld.coefficientGeneration = Publish開始時点の bank.generation
//   Learner は独立 Authority のため、Publish中に generation が進んでも RuntimeWorld は
//   Publish開始時点の値を保持する。これは正常動作でありバグではない。
//
//   NOTE: bankIndex と generation は lock-free snapshot で取得（個別 atomic load）。
//   完全な transactional snapshot は保証しない。RuntimeWorld.coefficient は診断・投影用途。
//   実際の係数適用は Audio Thread の generation tracking が行う。
const int bankIndex = convo::consumeAtomic(
    engine.currentAdaptiveCoeffBankIndex, std::memory_order_acquire);
worldOwner->coefficient.adaptiveCoeffBankIndex = bankIndex;

// ★ ISR Runtime 契約: RuntimeWorld 構築は accessor の副作用に依存しない。
//   明示的な範囲チェックにより純粋な Projection を保証する。
//   kNumAdaptiveCoeffBanks は AudioEngine.h に定義（180 = 10×3×6）。
//   RuntimeBuilder からは AudioEngine::kNumAdaptiveCoeffBanks で参照可能。
if (bankIndex >= 0 && bankIndex < static_cast<int>(AudioEngine::kNumAdaptiveCoeffBanks))
{
    const auto& bank = engine.getAdaptiveCoeffBankForIndex(bankIndex);
    worldOwner->coefficient.adaptiveCoeffGeneration = convo::consumeAtomic(
        bank.generation, std::memory_order_acquire);
}
```

**補足**:

- 明示的な `bankIndex < kNumAdaptiveCoeffBanks` チェックにより、clamp 実装への依存を排除し、
  Authority Projection を純粋に保つ（ISR Runtime 推奨パターン）
- 補足: `kNumAdaptiveCoeffBanks` = `kAdaptiveNoiseShaperSampleRateBankCount * kAdaptiveBitDepthCount * kLearningModeCount` = 180
- `eqCoeffHash` は現状維持（Publish世界で正しく設定されている）

### 3.2 テストの確認

**ファイル**: `src/tests/RuntimeWorldAuthorityProjectionTests.cpp` (L136-138)

現在のテストは以下を期待している:

```cpp
"worldOwner->coefficient.adaptiveCoeffBankIndex = engineState.adaptiveCoeffBankIndex;"
"worldOwner->coefficient.adaptiveCoeffGeneration = engineState.adaptiveCoeffGeneration;"
```

しかし現在の RuntimeBuilder は `engineState` ではなく直接 atomic 読取を使用する設計に
変更されている（[C4996 fix] コメント参照）。

テストは代入パターンを確認する。変数名非依存としつつも、コメントのみで
pass しないよう `coefficient.` プレフィックスを含める:

```cpp
// 代入パターンを確認（変数名非依存だが coefficient. スコープは確認）
"coefficient.adaptiveCoeffBankIndex"
"coefficient.adaptiveCoeffGeneration"
```

注意: 将来 `projectionGeneration` 等にリファクタリングされた場合は
テストも追従すること。

## 4. 修正の影響

### 4.1 直接影響を受ける箇所

| 箇所 | 現在 | P1後 | 影響 |
|------|------|------|------|
| `semanticHash.coefficientHash` | 定数（常に同じ値） | Publish開始時点の係数状態を反映 | **改善** |
| `[COEFF_AUTH] worldGen` | 常に 0 | Publish開始時点の bank.generation | **改善** |
| `AudioEngine.h L2318-2319` (buildAudioThreadProcessingState) | worldGen=0 から読み取り | worldGen=実値 から読み取り | 最初の1ブロックのみ影響（係数再適用なしになります） |
| `RuntimeWorldAuthorityProjectionTests.cpp` | テスト期待値と実コード乖離 | テスト期待値に合致 | **改善**（テストパス） |

### 4.2 注意: projectionLag は残る（正常）

P1 修正後も以下の lag は正常に発生し得る:

```text
RuntimeWorld (Publish時点) = bankGen = 10
  ↓ 5秒後
Live bankGen = 15  (learner が5世代進んだ)
  ↓
[COEFF_AUTH] worldGen=10 bankGen=15 lag=5  ← 正常
```

これは **異常ではない**。

#### 単調性に関する注意

理想的には `worldGen <= liveGen` が望ましいが、**厳密には保証できない**。
理由:

- `bankIndex` と `generation` は個別の atomic load で取得される
- Publish 完了後に Learner が bank を切り替え、Timer が別の bank を読む可能性がある
- ごく短時間の transient として `worldGen > liveGen` が一瞬観測され得る

判定基準:

```text
通常は worldGen <= liveGen
worldGen > liveGen が継続的に観測される場合のみ異常
```

#### coefficientHash と payloadHash の責務分離

P1 後、semanticHash の各サブハッシュは以下の Authority を表現する:

```text
payloadHash   = Structural Authority（BuildInput: SR, blockSize, OS設定等）
              → 係数変更では変化しない（設計として正しい）

coefficientHash = Adaptive Authority Projection（bankIndex, generation, eqCoeffHash）
                → P1 後、係数変更を検出するようになる

両者は独立して変化する
```

この分離により、`coefficientHash が変わったから rebuild しよう` という
誤った設計に戻ることを防止する。

### 4.3 coefficientHash の動作（P1.5 監査結果）

`coefficientHash` 生成コード（`RuntimeBuilder.cpp L417-419`）:

```cpp
semanticHash.coefficientHash =
    (adaptiveCoeffBankIndex + 0x100)
    ^ (adaptiveCoeffGeneration << 8)
    ^ (eqCoeffHash << 16);
```

P1.5 監査の結果、**3フィールドすべて**が `coefficientHash` に入力されていることを確認:

| フィールド | 使用 | P1前の状態 | P1後の状態 |
|-----------|------|-----------|-----------|
| `adaptiveCoeffBankIndex` | ✅ XOR + 0x100 offset | 常に0 → 定数寄与 | **実値に** |
| `adaptiveCoeffGeneration` | ✅ 左シフト8 | 常に0 → 定数寄与 | **実値に** |
| `eqCoeffHash` | ✅ 左シフト16 | 正しく設定 | 正しく設定（影響なし） |

**結論**: P1 後、`coefficientHash` は係数変更を正しく検出するようになる。
入力不足はない。

## 5. ログ名称の変更（推奨）

現在の `[COEFF_AUTH] divergence=` は「異常値」を連想させる。
P1 後は以下のように変更することが望ましい:

**変更前**:

```
[COEFF_AUTH] worldGen=0 bankGen=10 divergence=10
```

**変更後**:

```
[COEFF_AUTH] worldGen=10 bankGen=15 lag=5
```

変更箇所: `src/audioengine/AudioEngine.Timer.cpp` 内の `[COEFF_AUTH]` ログ。
`divergence` → `lag` に名称変更し、`worldGen=0` が改善されたことを確認できるようにする。

## 6. 検証項目

| # | 項目 | 方法 |
|---|------|------|
| 1 | RuntimeWorld.coefficient が Publish開始時点の bank 値を投影すること | `[COEFF_AUTH] worldGen` が 0 以外の値になることを確認 |
| 2 | `worldGen <= liveGen` の単調性が維持されること | 長時間ログで `worldGen > liveGen` が発生しないことを確認 |
| 3 | Learner 稼働中は projectionLag が発生するが異常ではないこと | ログで lag>0 が学習中に発生することを確認（正常動作） |
| 4 | semanticHash.coefficientHash が係数変更を検出すること | 係数更新前後で hash が変化することを確認 |
| 5 | テスト `RuntimeWorldAuthorityProjectionTests` が通過すること | 単体テスト実行 |
| 6 | CI compliance が通過すること | `check-list-compliance.ps1` 実行 |

## 7. 変更ファイル一覧

| ファイル | 変更内容 | リスク |
|---------|---------|--------|
| `src/audioengine/RuntimeBuilder.cpp` | coeff projection 追加（〜5行） | 低 |
| `src/audioengine/AudioEngine.Timer.cpp` | divergence→lag 名称変更（任意） | 低（診断ログのみ） |
| `src/tests/RuntimeWorldAuthorityProjectionTests.cpp` | 期待値パターン更新 | 低（テストのみ） |

## 8. 補足: なぜ divergence=0 を目標にしないのか

ISR Runtime の原則:

```
RuntimeWorld = Publish時点のスナップショット
AdaptiveCoeffBank = Live Authority（独立更新）
```

このため、Publish 後に Learner が係数を更新すると当然 generation が進む。
P1 の目的は「RuntimeWorld を完全に現在と一致させること」ではなく
「Publish 時点の正しい値を投影すること」である。

P1 後の正しさの条件:

```
worldGen <= liveGen    （単調性）
worldGen ≠ 0           （改善: 現在は常に0）
```
