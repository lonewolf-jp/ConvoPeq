# B-2 参照寿命保証の確認資料（2026-11-01）

- 状態: **確認資料のみ。製品コードは変更していない**（実装は Owner HOLD）
- 対象: `convDiagBuffer`（Convolver）と `eqDiagBuffer`（EQ）の**両方**
- 基準: `main` = `origin/main` = `6bb7c8d5`（ソースは `fd92c6e7` と同一）
- 目的: Owner 指示の 6 確認項目（B-2-1 / B-2-2）に、コード上の根拠で回答する

## 0. 結論（先に）

1. **音声コールバックの静止は証明できる。**
   ハーネスは `stopAudioOnly()` で **join**、製品は JUCE の `removeAudioCallback` が
   **`audioCallbackLock` による相互排除**で「実行中の callback が無い」ことを保証する。
2. **しかし、真の穴は別にある。** `ConvolverProcessor` は
   **エンジンに属さないインスタンス（テストのローカル変数）**からも `process()` が呼ばれ、
   その場合もグローバル `convDiagBuffer` に書き込む。
   エンジン破棄後は、このグローバルが**解放済みメモリ**を指す。
   **B-2（インスタンス所有）は、この経路を `sink_ == nullptr` にして消す。**
3. **detach 後の「load 済みポインタの使用」は、静止を先に確定させることで閉じる。**
   atomic 化だけでは閉じない（Owner 指摘どおり）。

## 1. 追加の実測（今回判明した決定的な事実）

### 1.1 リングバッファは配列をインライン保持する

`src/LockFreeRingBuffer.h:28`:

```cpp
alignas(64) T buffer[Capacity];          // ← ヒープポインタではなくインライン配列
alignas(64) std::atomic<size_t> writeIndex{0};
alignas(64) std::atomic<size_t> readIndex{0};
```

したがって書き込みアドレスは常に `リング先頭 + (writeIndex & 511) * sizeof(T)` であり、
**インデックスが壊れても範囲外にはならない**（0〜511 の剰余）。
**アクセス違反になるのは「リング先頭ポインタ自体が不正」な場合だけ**である。

### 1.2 クラッシュ命令の静的確認（ダンプの RVA を逆アセンブル）

RVA `0x253E21A`（ImageBase `0x140000000` → VMA `0x14253E21A`）:

```asm
14253e1ea: movq  0xb000(%rdi), %rdx     ; writeIndex（rdi + 45,056）
14253e1f1: movq  0xb040(%rdi), %rcx     ; readIndex （rdi + 45,120）
14253e1f8: movq  %rdx, %rax
14253e1fb: subq  %rcx, %rax
14253e1fe: cmpq  $0x200, %rax           ; 満杯判定
14253e204: jae   0x14253e252
14253e206: movq  %rdx, %rax
14253e209: andl  $0x1ff, %eax           ; w & 511
14253e20e: imulq $0x58, %rax, %rcx      ; × 88（sizeof(DiagEvent)）
14253e212: vmovups 0x140(%rbp), %ymm0   ; event の先頭 32 バイト
14253e21a: vmovups %ymm0, (%rcx,%rdi)   ; ★ ここでフォールト（書き込み）
```

- `%rdi` = **リング先頭（= グローバル `convDiagBuffer` の値）**
- フォールト時の書き込み先 = `0x1EC041AC800`
- インデックス項は最大 `511 × 88 = 44,968` なので、**`%rdi` 自体が `0x1EC041AC800` 近傍**であった
- `0x1E_xxxx_xxxx` は Windows のヒープ（大きなブロック）で使われるアドレス域と整合する
  → **解放済みエンジンのヒープを指したまま書き込んだ**（= UAF）と整合する

**これは「SPSC 違反でインデックスが壊れた」話ではない。ポインタの寿命の話である。**
（Owner の「atomic 化だけで解決したと判断しない」という指摘と一致する）

### 1.3 エンジンに属さない `ConvolverProcessor` が実在する

`src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp` には
**ローカル変数としての `ConvolverProcessor` が 23 個**あり、そのうち **`process()` 呼び出しが 14 箇所**ある
（例: `:719`, `:731`, `:1447`, `:1460`, `:1551`, `:1575`, `:1585`, `:1590`, `:1615`, `:1654`,
`:1734`, `:1991`, `:2105`, `:2201`）。

これらはエンジンに所有されないが、`ConvolverProcessor::process()` は
`#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` の下で**グローバル `convDiagBuffer` に書き込む**。

```
[テスト A] ハーネス生成 → エンジン A が setConvDiagBuffer(&A.diagBuffer)
[テスト A] 終了 → エンジン A 破棄 → &A.diagBuffer は解放（グローバルは A を指したまま）
[テスト B] ConvolverProcessor conv;  conv.process(blk);
             → グローバル経由で「解放済みの A」へ push
             → ヒープが再利用/デコミットされていればアクセス違反
```

**間欠性の説明**: 解放済みヒープがまだコミットされていれば書き込みは成功し（静かな破壊）、
再利用・デコミットされたときに初めて落ちる。したがって「毎回」ではなく 8.3% 程度になる。

## 2. B-2-1: オーディオコールバックの停止要求から join / 終了確認までの実経路

### 2.1 ハーネス（クラッシュが観測された構成）

| 段 | 実装 | 根拠 |
|---|---|---|
| 1 | 終了要求 | `running_ = false`（`exchange`、acq_rel） |
| 2 | **join** | `audioThread_.join()`（`AudioEngineHarness.cpp:59-66`） |
| 3 | 破棄 | `~AudioEngineHarness` が `stop()` を呼び、その後 `unique_ptr<AudioEngine>` が破棄される（`AudioEngineHarness.cpp:15-18`） |

**join は engine 破棄より前**である（`stop()` の中で完結）。
`abandonEngine()`（`AudioEngineHarness.cpp:85-89`）は `engine_.release()` で**破棄しない**ため、
この経路ではライフタイム問題は起きない。

### 2.2 製品（JUCE 音声デバイス）

| 段 | 実装 | 根拠 |
|---|---|---|
| 1 | コールバック解除 | `~MainWindow` step 7 `audioDeviceManager.removeAudioCallback(&audioProcessorPlayer)`（`MainWindow.cpp:1147`） |
| 2 | **実行中 callback の終了保証** | JUCE の `removeAudioCallback` は `ScopedLock sl(audioCallbackLock)` の下で登録を外す（`juce_AudioDeviceManager.cpp:1065-1076`）。音声スレッドも callback 全体を同じ lock で囲む（同 `:1083-1090`）。**lock による相互排除＝remove が返った時点で実行中の callback は無い** |
| 3 | デバイス閉鎖 | step 8 `closeAudioDevice()`（`MainWindow.cpp:1151`） |

### 2.3 `audioCallbackActiveCount == 0` の保証範囲と限界（B-2-1 の要求）

| 保証すること | 保証しないこと |
|---|---|
| 「その瞬間に**カウント対象のコールバックが実行されていない**」 | 「停止要求後に**新しいコールバックが開始されない**」 |
| カウンタの増減は callback 本体を囲む（`AudioBlock.cpp:93/98`、`BlockDouble.cpp:88/93`） | 「**別経路**（メッセージスレッド等）から同じバッファが参照されない」 |

**したがって順序は次のように固定する**（カウンタは「最後の確認」にのみ使う）:

```
① 新規開始の抑止: removeAudioCallback（製品）/ running_=false + join（ハーネス）
② 実行中 callback の終了: join（ハーネス）または audioCallbackLock（製品）
③ 補助確認: audioCallbackActiveCount == 0        ← 単独では不十分。①の後でのみ意味を持つ
④ detachDiagSink()（全生産者）
⑤ DiagSink の破棄（エンジンのメンバ破棄）
```

## 3. B-2-2: 全 DiagSink 生産者の列挙と停止・終了保証

製品コードの `DiagCategory` 代入箇所は **14**。バッファ別に整理する。

| # | 生産者（ファイル:行） | カテゴリ | バッファ | 実行スレッド | 停止・終了保証 |
|---|---|---|---|---|---|
| 1 | `AudioBlock.cpp:256` | CpuMig | `diagBuffer`（**メンバ参照**） | audio callback | ① join / lock |
| 2 | `AudioBlock.cpp:290` | CallbackSequence | 同上 | 同 | 同 |
| 3 | `AudioBlock.cpp:682` | DspTiming | 同上 | 同 | 同 |
| 4 | `AudioBlock.cpp:739` | CallbackStage | 同上 | 同 | 同 |
| 5 | `BlockDouble.cpp:243` | CpuMig | 同上 | 同 | 同 |
| 6 | `BlockDouble.cpp:280` | CallbackSequence | 同上 | 同 | 同 |
| 7 | `BlockDouble.cpp:642` | DspTiming | 同上 | 同 | 同 |
| 8 | `BlockDouble.cpp:691` | CallbackStage | 同上 | 同 | 同 |
| 9 | `DSPCoreFloat.cpp:79` | EqTime | **`eqDiagBuffer`（グローバル）** | audio callback | 同 |
| 10 | `DSPCoreDouble.cpp:39` | EqTime | 同上 | 同 | 同 |
| 11 | `DSPCoreDouble.cpp:647` | AnsSwitchTime | 同上 | 同 | 同 |
| 12 | `DSPCoreIO.cpp:435` | AnsSwitchTime | 同上 | 同 | 同 |
| 13 | `ConvolverProcessor.Runtime.cpp:767` | StereoConvTime | **`convDiagBuffer`（グローバル）** | **audio callback またはテストスレッド** | エンジン所有なら ①。**スタンドアロンは保証外** |
| 14 | `ConvolverProcessor.Runtime.cpp:852` | ConvTime | 同上 | 同上 | 同上 |

### 3.1 重要な切り分け

- **#1〜#8 は元から安全**（メンバ参照。グローバルを経由しない）。
- **#9〜#12（eq）はエンジン所有の DSP コアからのみ**呼ばれる（audio callback 内）。
- **#13〜#14（conv）だけが、エンジン非所有インスタンスからも呼ばれる**（§1.3）。
  **これが唯一の「停止で止められない生産者」である。**
- `uiConvolverProcessor` は `process()` が**一度も呼ばれない**（`uiConvolverProcessor.process` の呼び出し 0 件）
  → 生産者ではない。

### 3.2 各生産者の「参照しなくなったこと」の確認方法

| 生産者 | 確認方法 |
|---|---|
| #1〜#8 | メンバ参照なので、破棄と同時に参照が消える（グローバル無し） |
| #9〜#12 | ① + ② の後、audio callback は実行されない → push は発生しない。`audioCallbackActiveCount == 0` で補助確認 |
| #13〜#14（エンジン所有） | 同上 |
| #13〜#14（スタンドアロン） | **B-2 では `sink_ == nullptr`** のため push 自体が発生しない（設計で消す） |

## 4. detach 後に「取得済みポインタ」を使用する競合が残らないことの説明

**競合の形**:

```
producer: p = sink_.load(acquire)        // p != nullptr
engine:   detach (store nullptr); 破棄   // p が指す先が消える
producer: p->buffer->push(...)           // ← UAF
```

**閉じ方（2 段）**:

1. **detach より前に、全生産者の実行が終了している**ことを §2〜§3 の手順で確定する。
   したがって「detach と同時に push 中の生産者」は存在しない。
   → load と use の間隔に他スレッドが入り込む余地が無い。
2. detach は `release`、生産者は `acquire` で読む（見える順序の保証）。

**atomic 化だけでは足りない理由**: 1 を欠くと、load 済みの p を使う生産者が残る。
atomic は「値の一貫性」しか保証せず、「p の指す先が生存していること」は保証しない。

**残余のケース（設計で明示的に潰す）**: 将来、エンジン非所有の生産者が増えた場合は
`sink_` を attach しない（nullptr のまま）ことで同じ保証を維持する。
**attach は「所有エンジンだけが行う」という規則を型と API で強制する**（`DiagSink` を
エンジンの private メンバにし、`attach` をエンジンからのみ呼ぶ）。

## 5. conv と eq の同一保証（B-2-2 の要求 6）

| 観点 | conv | eq |
|---|---|---|
| バッファ所有 | `AudioEngine::diagBuffer`（メンバ） | 同じ（**同一バッファを共有**） |
| グローバル | `convDiagBuffer`（`ConvolverProcessor.Runtime.cpp:91`） | `eqDiagBuffer`（`DSPCoreFloat.cpp:31`） |
| エンジン所有生産者 | `convolver`（`AudioEngine.h:894`） | DSPCoreFloat/Double/IO |
| 停止保証 | ① join / lock | 同 |
| エンジン非所有生産者 | **あり**（テストのスタンドアロン 23 個・process 14 箇所） | なし（DSPCore は内部型） |
| B-2 適用後 | `sink_` は所有エンジンのみが attach。非所有は nullptr | 同 |

**両方とも「所有エンジンが attach し、破棄前に静止を確定して detach する」という同一手順で閉じる。**

## 6. 実装前に残る確認事項（実装フェーズでコードを書きながら確定）

1. `attachDiagSink` の呼び出し位置（`Init` か `prepareToPlay` か）と**冪等性**
   （`prepareToPlay` は複数回呼ばれる経路がある）。
2. DSPCore 群はエンジンごとに生成・破棄されるが、`DSPCore` の**生成タイミング**が
   `attach` より後になる場合の扱い（生成時に attach するか、エンジン経由で引くか）。
3. 診断 ON の ASan 構成（V4）を CI に追加するか（CI 定義の変更承認が必要）。

## 7. この資料で言えること・言えないこと

| 言えること | 言えないこと |
|---|---|
| 音声コールバックは join / lock で静止を確定できる | 「他の未知のスレッドが process() を呼ばない」ことの完全な保証（コード全域の動的証明はしていない） |
| スタンドアロン conv がグローバル経由で書き込む事実（14 箇所） | 今回のクラッシュがその経路 **だけ**で起きたことの断定（ダンプは 1 件） |
| 破棄前に静止を確定すれば、detach 後の load 済みポインタ使用は起きない | 負荷・配置によって発生率が変わる理由の完全な説明 |
| B-2 適用後、非所有生産者は push しない | 実装後の回帰結果（未実施） |

**したがって、B-2 の実装承認に必要な「寿命保証の設計」は成立していると判断する。
ただし実装は Owner の HOLD を維持し、コード変更は行っていない。**
