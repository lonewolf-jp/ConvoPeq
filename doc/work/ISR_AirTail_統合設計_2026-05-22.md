# Air Absorption / Layer Tail Contouring - Immutable Snapshot Runtime 統合設計（ConvoPeq）

作成日: 2026-05-22
対象: `ConvolverProcessor` + `MKLNonUniformConvolver` を ISR（Immutable Snapshot Runtime）へ完全整合させる

---

## 1. 目的

`Air Absorption` と `Layer Tail Contouring` を、既存の可変パラメータ駆動ではなく、
**Immutable Snapshot Runtime（以下 ISR）** の publish 取引に統合する。

達成条件:

1. SoT は `GlobalSnapshot -> BuildSnapshot -> RuntimeWorld` の単一路
2. Audio Thread は read-only（consume のみ）
3. publish 後に tail モード関連の mutable 書き換えを行わない
4. rebuild / crossfade / retire を ISR transaction 境界内で完結する

---

## 2. 非交渉 Invariant

1. Audio Thread で禁止:
   - `tailMode/tailStart/tailStrength/tailL1L2` の決定ロジック実行
   - パーティション再構築
   - FFTプラン再初期化
   - lock/alloc/log/UI/logging
2. `FilterSpec` は publish 前に完全確定
3. `MKLNonUniformConvolver::SetImpulse()` は Message/Builder 相当スレッド専用
4. `Get()` は「確定済み layer gain」と「確定済み tail output」を合成するだけ

---

## 3. SoT モデル（責務境界）

### 3.1 正本（SoT）

- UI入力: `PendingParams`
- publish入力: `BuildSnapshot`
- runtime反映: `convo::FilterSpec` + `RuntimeWorld`（publish payload）

### 3.2 パラメータ責務

- `tailMode`: 挙動クラス選択（Air / Layer / Bypass）
- `tailStartSec`: L0 と L1境界決定
- `tailStrength`: tail の主強度
- `tailL1L2Multiplier`: partition 階層倍率

### 3.3 導出値（非SoT）

以下は **build時導出** として扱い、SoTへ保存しない:

- `layer1Gain`, `layer2Gain`
- Air モードの HF damping curve（周波数ゲイン配列）
- L0/L1/L2 の実 partition 長

---

## 4. 統合アーキテクチャ

```mermaid
flowchart LR
  UI[UI Tail Params] --> P[PendingParams]
  P --> S[BuildSnapshot capture]
  S --> F[FilterSpec materialize]
  F --> B[SetImpulse build]
  B --> W[RuntimeWorld publish]
  W --> A[Audio Thread consume]
  A --> O[Get(): read-only mix]
```

設計ポイント:

- `FilterSpec` materialize でモード別プロファイルを確定
- `SetImpulse` 内で IR 分割・周波数整形を確定
- Audio Thread は `m_tailLayerGain[]` を読むだけ

---

## 5. モード別アルゴリズム仕様（ISR向け）

### 5.1 Air Absorption

方針:

- Early（L0）保持、Late（L1/L2）ほど減衰
- L1/L2 の周波数領域IRに HF damping を事前焼き込み

処理:

1. `tailStartSec` は Air 用下限で clamp
2. `tailL1L2Multiplier` は Air 用の安定領域へ clamp
3. `layer1Gain/layer2Gain` を `tailStrength` から導出
4. L1/L2 の `irFreqDomain` へ高域減衰カーブを乗算
5. deinterleave を再生成（SoA整合）

### 5.2 Layer Tail Contouring

方針:

- tail 輪郭を立てるため、L1 を主軸、L2 は補助に設定
- 既存の強化下限（start/strength/multiplier）を維持

処理:

1. Layer モード下限を適用
2. `layer1Gain > layer2Gain` となる輪郭プロファイルを導出
3. `Get()` で層別ゲイン合成

### 5.3 Bypass

- `tailEnabled=false` と同等
- `layer1Gain/layer2Gain=0`
- L1/L2 合成寄与なし

---

## 6. RuntimeWorld への統合ポイント

### 6.1 publish 前（Builder/Message）

- `BuildSnapshot` から `FilterSpec` を構成
- `SetImpulse(..., &spec)` で runtime engine を構築
- 成功した engine のみ publish

### 6.2 publish 後（Audio Thread）

- `RuntimePublishWorld` から active/fading を acquire
- `StereoConvolver::process()` -> `Get()` で read-only 合成
- tail モード変更は次回 publish まで反映しない

### 6.3 retire

- 旧 engine は retire 経路へ移譲
- Audio Thread は旧 engine の寿命管理に関与しない

---

## 7. 変更契約（API/構造）

### 7.1 必須契約

1. `setTail*` 系は `PendingParams` の更新のみ
2. `captureBuildSnapshot()` で tail パラメータを輸送
3. `FilterSpec` に mode/start/strength/multiplier を保持
4. `SetImpulse()` 内で `m_tailLayerGain[]` を確定

### 7.2 禁止契約

1. Audio Thread で `tailMode` 分岐による rebuild 判定
2. `Get()` 内でモード依存の係数再計算
3. publish 後 engine への tail パラメータ再注入

---

## 8. 失敗時挙動（transaction safety）

1. `SetImpulse` 失敗時:
   - publish 中止
   - 現行 runtime 継続
2. `FilterSpec` 不整合時:
   - clamp + fail-safe（Bypass相当）
3. shutdown 中:
   - rebuild 要求を拒否
   - 既存 runtime のみ安全停止

---

## 9. 検証項目（統合完了判定）

### 9.1 機能

- TailMode 切替（Air/Layer/Bypass）で音切れなく挙動変化
- Air: 高域テールが段階的に短くなる
- Layer: L1 輪郭が強く、L2 は補助的

### 9.2 ISR整合

- Audio Thread で tail 再計算なし
- publish 後 mutable 更新なし
- retire/reclaim が非RTで完結

### 9.3 回帰

- Debug/Release build pass
- Strict Atomic Dot-Call Scan pass
- list compliance fail=0 維持

---

## 10. 段階移行プラン（最小リスク）

1. Phase-A: `FilterSpec` / `SetImpulse` にモード別導出ロジック集約
2. Phase-B: `Get()` を層別ゲイン合成へ統一
3. Phase-C: runtime publish world 経路以外の tail 反映経路を削除
4. Phase-D: 検証スクリプトに tail mode 回帰観点を追加

---

## 11. 受入基準（Done）

- [ ] Air/Layer/Bypass が ISR publish 単位で反映される
- [ ] Audio Thread は read-only 合成のみ
- [ ] tail mode 変更時の反映境界が publish に一致
- [ ] build/scan/compliance が全て green

---

## 12. 既存コードへのマッピング（実装済み整合点）

本設計に対し、既存コードでは以下の形で整合が取れている。

- `FilterSpec` に tail mode/start/strength/multiplier を保持
- `SetImpulse()` 内でモード別 clamp/導出を実施
- `m_tailLayerGain[]` を build 時に確定し、`Get()` で read-only 使用
- Air モードで L1/L2 の周波数領域IRへ HF damping を事前適用

このため、以後の拡張は `BuildSnapshot -> FilterSpec -> SetImpulse` 境界に限定して実施できる。
