# ConvoPeq ソースコード調査報告書（Part 6）

`ISRLifecycle.cpp/.h` のレビュー結果です。Part 5で見つけた「安全網は実装されているが接続されていない」パターンが、別のサブシステムでも再現していることを確認しました。

---

## 0. 今回のサマリ

| # | 重大度 | ファイル | 概要 |
|---|--------|----------|------|
| 8 | **Medium**（現状クラッシュ等は起きないが、想定されている安全機構が機能していない） | `ISRLifecycle.cpp/.h` | `LifecycleToken`（`epochId`+`expectedPhase`）は`enterPrepare`/`enterAudioCallback`/`enterRelease`の3箇所で発行されるが、対になる`leavePrepare`/`leaveAudioCallback`/`leaveRelease`はいずれも受け取った`token`引数を一切使用しない。ヘッダのコメントで明記されている受入条件「LIF-5: callback中runtimeVersion変化なし」「LIF-6: callback中DSP generation変化なし」を実際に検証するコードが存在しない。 |

---

## 1. 【Medium】LifecycleToken が発行されるだけで検証されていない

### 該当ファイル

`src/audioengine/ISRLifecycle.h`（`LifecycleToken`定義、ローカル31-35行目）
`src/audioengine/ISRLifecycle.cpp`（`enterPrepare`/`leavePrepare`, `enterAudioCallback`/`leaveAudioCallback`, `enterRelease`/`leaveRelease`）

### ヘッダのコメントが示す設計意図

```cpp
/**
 * LifecyclePhase ステートマシン runtime
 * JUCE callback の違反（overlap, late callback, etc.）を検出・abort する。
 *
 * 受入条件（LIF-1～LIF-6）:
 *  - LIF-1: prepareToPlay serialized
 *  - LIF-2: releaseResources は AudioRunning 中に呼べない
 *  - LIF-3: Releasing phase 中の publish 禁止
 *  - LIF-4: crossfade start は Prepared 以降のみ
 *  - LIF-5: callback 中 runtimeVersion 変化なし
 *  - LIF-6: callback 中 DSP generation 変化なし
 */
```

`LifecycleToken`は「callback入口でのepochトークン」（ヘッダのコメントより）として設計されており、`enterXxx()`が返した`epochId`を`leaveXxx(token)`で照合すれば、LIF-5/LIF-6（"入口から出口までの間にepoch/generationが変化していないか"）を検証できる構造になっています。

### 実際のコード（3ペアとも同型）

```cpp
LifecycleToken LifecycleIsolationRuntime::enterAudioCallback()
{
    ...
    transitionTo(LifecyclePhase::AudioRunning);
    uint64_t epochId = convo::consumeAtomic(epochCounter_, std::memory_order_acquire);
    return LifecycleToken{ epochId, LifecyclePhase::AudioRunning };   // ← epochIdを記録して返す
}

void LifecycleIsolationRuntime::leaveAudioCallback(LifecycleToken token)   // ← token を受け取るが…
{
    auto currentPhase = convo::consumeAtomic(phase_, std::memory_order_acquire);
    if (currentPhase != LifecyclePhase::AudioRunning) {
        std::abort();
    }
    transitionTo(LifecyclePhase::Prepared);
    // token.epochId は一切参照されない
}
```

`enterPrepare`/`leavePrepare`、`enterRelease`/`leaveRelease`も全く同じ構造で、`leave`側は`token`引数を受け取るだけで中身を一度も読みません（コンパイラの`-Wunused-parameter`が有効なら警告が出るはずのコードです）。

### 影響

- LIF-5/LIF-6として明文化されている「callback実行中にruntimeVersion/DSP generationが変化していないこと」の検証は、**現状は何も行われていません**。
- 呼び出し元（`AudioEngine.Processing.AudioBlock.cpp`/`BlockDouble.cpp`）を確認したところ、`lifecycleToken`はRAIIガード内に保持されるのみで、生成された`token`を使って何か比較検証をしている形跡もありませんでした。
- 現状、audio callback の phase 遷移そのもの（Prepared→AudioRunning→Prepared、および不正な遷移の`abort()`）は`transitionTo`/`validateTransition`により正しく機能しています。問題は**「callback実行中に何かがこっそり変化していないか」という、より踏み込んだ整合性検証**が未実装という点に限られます。
- Part 5で報告したRT Capability/Allocator Firewall（安全網は用意されているが接続されていない）と全く同じ形のギャップが、独立したサブシステムでも見つかったことになります。プロジェクト全体でこの「検証インフラは構築したが最後の照合ステップが未実装」というパターンが繰り返されている可能性があるため、体系的な洗い出しを推奨します。

### 検証根拠

- `enterPrepare`/`enterAudioCallback`/`enterRelease`の3関数全てで`LifecycleToken`が`epochId`付きで構築・返却されていることを確認。
- 対になる`leavePrepare`/`leaveAudioCallback`/`leaveRelease`の3関数全てで、受け取った`token`パラメータの参照が**本文中に一度も無い**ことをソースコードで直接確認（`grep`ではなく実装全文の目視確認）。
- `enterAudioCallback`/`leaveAudioCallback`が実際にAudio Thread本体（`AudioEngine.Processing.AudioBlock.cpp`/`BlockDouble.cpp`）から呼ばれていることも確認済み（Part 5のRT Firewallとは異なり、こちらは実際にRTパスへ配線されています。欠けているのは配線ではなく、token照合ロジックそのものです）。

### 推奨対応（パッチは提示しません — 何と照合すべきかの仕様確認が必要なため）

`leaveXxx(token)`側で、現在の`epochCounter_`（または該当するruntimeVersion/DSP generation相当の値）を`token.epochId`と比較し、不一致ならLIF-5/LIF-6違反として扱うロジックの実装を推奨します。ただし:

- 現状の`epochCounter_`は`Prepared`/`Released`遷移でのみインクリメントされる設計であり（`transitionTo`実装で確認済み）、「callback中のDSP generation変化」を検知する目的にそのまま使えるかは要検討です（DSP generationは`ISRDSPHandle`側の別カウンタで管理されているため、LIF-6の検証には`epochCounter_`ではなく`DSPHandleRuntime`側のgenerationとの突合が必要になる可能性があります）。
- そのため、どの変数を何と比較すべきかはアーキテクチャ全体の意図に依存すると判断し、今回はパッチ化を見送りました。

---

## 2. 検証したが問題なしと判断した箇所（追加分）

### 2.1 LifecyclePhase 状態遷移表（`validateTransition`）

`AudioRunning`状態への再入（re-entrant call）が`enterAudioCallback()`冒頭のガード条件では許容されているように見えましたが、実際には後続の`transitionTo(AudioRunning)`→`validateTransition(AudioRunning, AudioRunning)`が`AudioRunning`ケースで`to==AudioRunning`を許可リストに含めていないため、最終的に`std::abort()`で正しく検知されることを確認しました。一見冗長なガード構造ですが、機能上の欠陥はありません。

### 2.2 `transitionTo()`内の`std::chrono::high_resolution_clock::now()`

Audio Thread（`enterAudioCallback`/`leaveAudioCallback`経由）から呼ばれますが、Windows環境では`QueryPerformanceCounter`相当の非ブロッキング呼び出しにマップされ、既存コードの`convo::getCurrentTimeUs()`と同様の許容パターンと判断しました。

---

## 3. 調査範囲の更新

- `ISRLifecycle.cpp`/`.h`（合計約290行）: 精読完了。
- `ISRDSPHandle.cpp`: `retire`/`reclaim`/`quarantine`/`rollbackRegistration`/`quarantineSlot`/`isSlotInCrossfade`/`destroyQuarantineSlot`まで確認。`reclaim()`が非アトミックな`instance`書き込み後に`state`をpublishする点について、`resolve()`との理論的な競合可能性を検討しましたが、実際に両者が競合し得るかは呼び出し元のepoch/grace-period保証（未確認の`ISRRetire.cpp`等）に依存するため、断定は保留しています。
- `ISRDSPQuarantine.cpp`/`.h`（約250行）: 精読完了。`QuarantineEntry::quarantineEpoch`が実際にはタイムスタンプの複製であり、専用の値を持たない点に気づきましたが、この値を読む箇所が存在しないため実害なしと判断し、正式なバグとしては報告していません。

---

## 4. 次のステップ（提案）

これで約280KB中、Part 1〜6を通じて8件の指摘事項（確定4件、要確認/要検証4件）をまとめました。分量が増えてきましたので、次のいずれかを選べます:

1. このまま`ISRRuntimePublicationCoordinator.cpp`（ISR系最大の529行）や残りのISR系ファイルへ継続
2. 一旦ここまでの8件を統合したサマリ（マスターインデックス）を作成し、区切りとする
3. `EQProcessor.Coefficients.cpp`の係数計算式の数式検証など、ISR系以外の未消化項目に切り替える

ご指定がなければ1の方向で継続します。
