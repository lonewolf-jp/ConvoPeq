# BUG 9: EQProcessor シャドウ relaxed-only データ競合

- **発見日**: 2026-07-25
- **重要度**: Medium
- **影響**: NonRT→RT のパラメータ伝搬タイミングが未定義、一部の変数では二重書き込みによる競合

## 場所

| ファイル | 行 (NonRT 書き込み) | 行 (RT 書き込み) | 行 (RT 読み取り) |
|---------|---------------------|-------------------|-------------------|
| `EQProcessor.Core.cpp` | 592-596, 655-659 | — | — |
| `EQProcessor.Processing.cpp` | — | 434-436, 506, 517, 586-588, 919, 926, 1008, 1010, 1070-1072 | 415-417, 497, 509, 863, 1019 |

## 該当変数

| 変数 | 型 | NonRT 書き手 | RT 書き手 | RT 読み手 |
|------|-----|-------------|-----------|-----------|
| `rtBypassedShadow` | `std::atomic<bool>` | Core:592,655 | Proc:506,517,1008,1010 | Proc:497,509,1019 |
| `rtActiveStructureShadow` | `std::atomic<FilterStructure>` | Core:593,656 | Proc:919,926 | Proc:863 |
| `rtAgcCurrentGainShadow` | `std::atomic<double>` | Core:594,657 | Proc:436,586,1070 | Proc:417 |
| `rtAgcEnvInputShadow` | `std::atomic<double>` | Core:595,658 | Proc:434,587,1071 | Proc:415 |
| `rtAgcEnvOutputShadow` | `std::atomic<double>` | Core:596,659 | Proc:435,588,1072 | Proc:416 |

## 現象

すべてのシャドウ変数が `memory_order_relaxed` のみを使用して読み書きされている。
C++ メモリモデル上、relaxed 順序付けはスレッド間の happens-before を一切形成しない。

### 問題 1: NonRT→RT 一方向伝搬（すべての変数に共通）

NonRT（`syncBandNodeFrom`, `syncGlobalStateFrom`）が relaxed store で値を設定しても、
RT スレッドがその値を relaxed load で読んだときに最新値を観測できる保証がない。
x86/64 では StoreLoad の順序が強いため現実的な問題は稀だが、ARM64 では無視できない。

### 問題 2: 二重書き手競合（特に rtBypassedShadow）

`rtBypassedShadow` は **NonRT と RT の両方が書き込む**二重書き手変数：
- NonRT は同期時に目的のバイパス状態を書き込む
- RT はフェード完了後に現在の有効バイパス状態を書き込む

両者の relaxed store には順序制約がないため、NonRT の store と RT の store が
任意の順序で観測される可能性がある。

### 問題 3: AGC シャドウ（rtAgcEnvInputShadow 等）の実用上のリスク

AGC（自動ゲイン制御）のエンベロープ値は RT スレッドが毎ブロック更新する。
NonRT は同期時に初期値をシードするだけであるため、NonRT の relaxed store が
RT に見えなくても次ブロックで RT が再計算するため実害は限定的。
ただし、初期シード損失により、同期直後の AGC 状態遷移に一過性の値が発生する可能性がある。

## 影響評価

- **rtBypassedShadow**: ユーザーがバイパスをトグルした直後、RT が目的状態を認識できず
  一過性のオーディオグリッチ（想定：~1 ブロック分）が発生する可能性がある。
  フェード遷移中は `bypassFadeGain` LinearRamp が緩衝するため実害は軽微。
- **rtActiveStructureShadow**: 構造切り替え直後に一過性のミスマッチが発生する可能性がある。
- **rtAgc シャドウ群**: NonRT シードはベストエフォート。実害はほぼなし。

## 修正方針

**推奨**: 書き込み側に release、読み取り側に acquire を追加（標準的な HB 形成）。

```cpp
// NonRT（EQProcessor.Core.cpp）
rtBypassedShadow.store(syncedBypassed, std::memory_order_release);
rtActiveStructureShadow.store(syncedStructure, std::memory_order_release);
// 同様に他のシャドウも release に変更
```

```cpp
// RT（EQProcessor.Processing.cpp）
double envIn = rtAgcEnvInputShadow.load(std::memory_order_acquire);
// 同様に他の読み取りも acquire に変更
```

**代替案**: NonRT→RT の伝搬のみを release/acquire とし、RT→RT の再書き込みは relaxed のまま維持する。
この場合、RT の書き込み後、RT の次回読み取りはプログラム順序ですでに可視であるため relaxed で十分。
ただし、データ競合の観点からはすべての読み取りを acquire に統一するのが安全。

**注意**: `rtDeferredBandResetMask`（Core:598）は RT の書き込みのみ（Proc:505）なので relaxed で正しい。
