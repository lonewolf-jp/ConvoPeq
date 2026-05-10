# ConvoPeq TX-11〜TX-15 テスト手順書

## 目的

本手順書は、次世代アーキ移行におけるパラメーター網羅性テスト（TX-11〜TX-15）を実施するための運用資料である。
対象は以下。

1. Preset / DeviceSettings の保存復元対称性
2. Snapshot の等価判定感度
3. 内部 OS レート導出整合
4. 復元順序依存の安全性

## 事前条件

1. Debug build が成功していること
2. テスト入力素材（固定の音源ファイル）を1つ用意すること
3. ログ確認が可能であること
4. 既存設定ファイルのバックアップを取得済みであること

## 共通記録フォーマット

各 TX 実施ごとに以下を記録する。

1. 入力条件

- SR
- Block Size
- 既存設定ファイル version
- IR finalized 状態

1. 期待値

- SoT 値（AudioEngine getter）
- 派生値（例: processing sample rate）

1. 観測値

- 復元後 getter 値
- snapshot hash 変化有無
- rebuild 発火回数

1. 判定

- pass/fail
- 差分理由

## TX-11 Preset Roundtrip

### 目的

Preset 保存/読込の対称性を確認する。

### 手順

1. 任意の初期状態を作成する。

- processingOrder
- bypass
- inputHeadroomDb/outputMakeupDb
- oversamplingFactor/oversamplingType
- ditherBitDepth/noiseShaperType

1. Preset を保存する（xml）。
2. 初期状態を大きく変更する。
3. 2で保存した Preset を読込む。
4. 保存前と読込後の主要 getter 値を比較する。

### 合格条件

1. 保存対象キーの集合が一致する。
2. 保存前と読込後の getter 値が一致する。
3. 不要な二重 rebuild が発生しない。

## TX-12 DeviceSettings Roundtrip

### 目的

DeviceSettings の保存復元対称性を確認する。

### 手順

1. 以下を変更し saveSettings を実行する。

- ditherBitDepth
- noiseShaperType
- fixedNoiseLogIntervalMs/fixedNoiseWindowSamples
- oversamplingFactor/oversamplingType
- inputHeadroomDb/outputMakeupDb

1. アプリ再起動相当の手順で loadSettings を実行する。
2. 復元後 getter 値を保存時の値と比較する。

### 合格条件

1. 保存項目が復元される。
2. デバイス依存情報の復元失敗時に安全なフォールバックが機能する。

## TX-13 Snapshot Hash Sensitivity

### 目的

sampleRate/maxBlockSize 変化が snapshot 等価判定へ反映されることを確認する。

### 手順

1. 固定パラメーターのまま SR を変更する（例 48k -> 96k）。
2. snapshot createImpl の挙動を観測する。
3. block size も変更して同様に観測する。

### 合格条件

1. SR または block size 変化時に no-op 抑制されず新 snapshot が生成される。
2. 等価判定が false になることをログまたは観測値で確認できる。

## TX-14 Derived Rate Integrity

### 目的

内部 OS レート（processing sample rate）の導出整合を確認する。

### 手順

1. SR と oversamplingFactor の組み合わせを複数試験する。

- 48k with Auto
- 48k with 2x/4x/8x
- 96k with Auto
- 192k with Auto

1. getProcessingSampleRate の値を記録する。
2. 期待式（SR × actualFactor）と比較する。

### 合格条件

1. 全組み合わせで期待式と一致する。
2. Auto 時の factor 上限クランプが仕様通りである。

## TX-15 Restore Order Safety

### 目的

復元順序依存の不整合（headroom clamp 破綻、二重 rebuild）を検出する。

### 手順

1. processingOrder と eq/conv bypass を含む Preset を作成して保存する。
2. 現在状態を別モードに変更する。
3. Preset を読込む。
4. 復元直後の headroom/makeup/trim と rebuild 回数を確認する。

### 合格条件

1. mode/bypass 先行復元の後に gain 系が正しいクランプ値で反映される。
2. restore 中の不要な applyDefaults 上書きが発生しない。
3. restore 終端で rebuild 判定が1回に収束する。

## 実施後チェック

1. fail が1件でもあれば、失敗条件を nextgen_runtime_transition_design_jp.md の該当章へ追記する。
2. fail が復元順序由来の場合、まず requestLoadState の順序と guard 範囲を確認する。
3. fail が snapshot 由来の場合、SnapshotParams と SnapshotFactory の比較項目を確認する。

## 参照

1. doc/nextgen_runtime_transition_design_jp.md
2. src/audioengine/AudioEngine.StateIO.cpp
3. src/core/SnapshotParams.h
4. src/core/SnapshotFactory.cpp
5. src/DeviceSettings.cpp
