# B-2 参照寿命保証の確認報告（2026-11-01 第 3 部）

- 基準: `main` = `origin/main` = **`6bb7c8d5`**（ソースは `fd92c6e7` と同一）
- **製品コード・テストコードは変更していない。build / test / debugger / replay も実行していない**
- 詳細: `doc/residual-20261101/B2-LIFETIME-PROOF-20261101.md`、
  `doc/residual-20261101/A2-A3A-SOURCE-RECONCILIATION-20261101.md`

## 1. B-2-1: オーディオコールバックの停止・join 経路

| 構成 | 経路 | 保証 |
|---|---|---|
| **ハーネス**（クラッシュ観測構成） | `~AudioEngineHarness` → `stop()` → `stopAudioOnly()` が `running_=false`（acq_rel）→ **`audioThread_.join()`**（`AudioEngineHarness.cpp:38-66`） | **join はエンジン破棄より前**。`abandonEngine()` は破棄しない（`engine_.release()`）ため対象外 |
| **製品**（JUCE デバイス） | `~MainWindow` step7 `removeAudioCallback`（`MainWindow.cpp:1147`）→ step8 `closeAudioDevice`（`:1151`） | JUCE の `removeAudioCallback` は `ScopedLock(audioCallbackLock)` の下で登録を外し（`juce_AudioDeviceManager.cpp:1065-1076`）、音声スレッドも callback 全体を同じ lock で囲む（`:1083-1090`）。**lock による相互排除＝remove が返れば実行中 callback は無い** |

### `audioCallbackActiveCount == 0` の保証範囲と限界

| 保証する | 保証しない |
|---|---|
| その瞬間にカウント対象の callback が実行されていない（増減は `AudioBlock.cpp:93/98`・`BlockDouble.cpp:88/93` で本体を囲む） | **停止要求後に新しい callback が開始されないこと** |
| — | **別経路（メッセージスレッド・テストスレッド）から同じバッファが参照されないこと** |

**確定した順序**: ①停止（新規開始の抑止）→②join / lock（実行中の終了）→③カウント 0 の確認（補助）
→④`detachDiagSink()`→⑤破棄。**カウンタは③の補助にのみ使い、①の代替にしない。**

## 2. B-2-2: 全生産者の列挙と停止保証

製品コードの `DiagCategory` 代入は **14 箇所**。

| バッファ | 生産者 | 実行スレッド | 停止保証 |
|---|---|---|---|
| `diagBuffer`（**メンバ参照**） | `AudioBlock.cpp:256/290/682/739`、`BlockDouble.cpp:243/280/642/691` | audio callback | join / lock。**元からグローバル非経由で安全** |
| `eqDiagBuffer`（グローバル） | `DSPCoreFloat.cpp:79`、`DSPCoreDouble.cpp:39/647`、`DSPCoreIO.cpp:435` | audio callback | エンジン所有 DSP コア → join / lock |
| `convDiagBuffer`（グローバル） | `ConvolverProcessor.Runtime.cpp:767/852` | audio callback **またはテストスレッド** | エンジン所有は join / lock。**非所有は停止で止まらない** |

`uiConvolverProcessor.process` は**呼び出し 0 件**（生産者ではない）。

## 3. 決定的な発見: エンジン非所有の `ConvolverProcessor`

`ConvolverStateRoundTripTests.cpp` には**ローカルの `ConvolverProcessor` が 23 個**あり、
**`process()` 呼び出しが 14 箇所**ある（`:719`, `:731`, `:1447`, `:1460`, `:1551`, `:1575`, `:1585`,
`:1590`, `:1615`, `:1654`, `:1734`, `:1991`, `:2105`, `:2201`）。

これらはエンジンに所有されないが、`process()` は `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` の下で
**グローバル `convDiagBuffer` に書き込む**。エンジン破棄後は、このグローバルが**解放済みメモリ**を指す。

```
テスト A: ハーネス生成 → グローバル = &A.diagBuffer
テスト A: 終了 → エンジン A 破棄 → &A.diagBuffer 解放（グローバルは A を指したまま）
テスト B: ConvolverProcessor conv; conv.process(blk); → 解放済み A へ push
```

**B-2 では非所有インスタンスの `sink_` が `nullptr`** なので、この経路は**書き込み自体が発生しない**。

## 4. クラッシュ命令の静的確認（ポインタ寿命の裏付け）

`LockFreeRingBuffer.h:28` は `T buffer[Capacity]` を**インライン保持**するため、書き込みアドレスは
`先頭 + (writeIndex & 511) * 88` に限られ、**インデックス破壊では範囲外にならない**。
ダンプの RVA `0x253E21A` を逆アセンブルすると:

```asm
14253e1ea: movq 0xb000(%rdi), %rdx    ; writeIndex
14253e209: andl $0x1ff, %eax
14253e20e: imulq $0x58, %rax, %rcx    ; × 88
14253e21a: vmovups %ymm0, (%rcx,%rdi) ; ★ 書き込みでフォールト
```

書き込み先 `0x1EC041AC800` は `%rdi`（= `convDiagBuffer`）近傍であり、
`0x1E_xxxx_xxxx` は Windows の大きなヒープブロックのアドレス域と整合する。
**= 解放済みエンジンのヒープを指したまま書いた（UAF）**。SPSC 違反ではなく**ポインタ寿命の欠陥**である。

**間欠性の説明**: 解放済みヒープがコミットされたままだと書き込みは成功し（静かな破壊）、
再利用・デコミット時に初めて落ちる。したがって毎回ではなく 8.3% 程度になる。

## 5. detach 後に「取得済みポインタ」を使う競合が残らないこと

```
producer: p = sink_.load(acquire)      // p != nullptr
engine:   detach; 破棄                  // p の先が消える
producer: p->buffer->push(...)         // ← UAF
```

**閉じ方**: detach より前に全生産者の静止を §1・§2 で確定するため、**load と use の間に
他スレッドが入り込まない**。detach は release、生産者は acquire。加えて
**「attach は所有エンジンだけが行う」**規則（`DiagSink` を private メンバにし、API をエンジンに限定）で
将来の非所有生産者の混入も防ぐ。

**atomic 化だけでは閉じない**: atomic は値の一貫性しか保証せず、指す先の生存は保証しない。

## 6. conv / eq の同一保証

| 観点 | conv | eq |
|---|---|---|
| バッファ | `AudioEngine::diagBuffer`（メンバ、**eq と共有**） | 同 |
| グローバル | `convDiagBuffer`（`ConvolverProcessor.Runtime.cpp:91`） | `eqDiagBuffer`（`DSPCoreFloat.cpp:31`） |
| エンジン所有生産者 | `convolver`（`AudioEngine.h:894`） | DSPCoreFloat/Double/IO |
| 非所有生産者 | **あり**（テストの 23 個） | なし（DSPCore は内部型） |
| B-2 後 | 所有のみ attach、非所有は nullptr | 同 |

**両方とも「所有エンジンが attach し、破棄前に静止を確定して detach する」同一手順で閉じる。**

## 7. A2 → A3-a の最新ソース照合（要点）

| 項目 | 実測 |
|---|---|
| assert の位置 | `ISRDSPHandle.cpp:527-529`（**CAS `:562-572` より前**） |
| deferred 計上 | `:551-559`（pin gate で `deferredQuarantineDestroy_ += 1`） |
| retry の範囲 | `:598-610`（**Quarantined な全スロット**を走査） |
| 呼び出し元 | `Commit.cpp:682`、`ReleaseResources.cpp:468`（destroy）／`ReleaseResources.cpp:450`、`Retire.cpp:55`・`:89`（retry） |
| **ゲート契約** | `deferredQuarantineDestroyCount()` は `Retire.cpp:44`・`:72` の**早期 return 条件**。A3-a 後も「保留中のスロット数」を返さないと**リークする** |
| A2 | 述語を **CAS 成功時の不変条件**として保持し、**`Quarantined` への逆行**を検出する assert を追加（削除ではない） |
| A3-a | `uint64`×4 のビットマスク（`MAX_DSP_SLOTS=256`）でスロット単位マーク。tick はマーク済みのみ |
| 順序 | **A2 → A3-a** |

## 8. D-01 CLOSE

**旧分析の誤り**: `D01-STG8-ANALYSIS-20261101.md` §2 RC-1 の「getter の呼び出し元は 0」
「その分岐は現存しない」は誤り。`PublicationAdmission.cpp:51-56` にフックは存在し、
`a12e05f2` でも存在していた。

**製品経路とテスト専用フックの区別**:

| 観点 | 実測 |
|---|---|
| 製品の呼び出し経路 | `RuntimePublicationOrchestrator.cpp:47` の `admission_.evaluate(...)` |
| 製品の判定条件 | フックは `#if defined(CONVOPEQ_UNIT_TESTS)` の内側＝**コンパイル対象外**。製品は `hasFadingRuntimeInWorld`（`AudioEngine.h:3621`）のみ |
| 製品での書き手 | テスト専用クラスのみ（`CONVOPEQ_ALL_SOURCES` に `src/tests` は 0 件） |

**結論: 追加 2 行は適用しない（D-01 CLOSE）。** 当初目的（STG-8-D2 の retention 検証）は
`CONVOPEQ_UNIT_TESTS=1` のテストで既に達成されている。

## 9. 成果物と遵守

- Dropbox: `investigation-20261101/` に **37 ファイル・ALL MATCH**
  （本報告の 2 資料を追加）
- `ConvoPeq.md`（`6bb7c8d5` の blob・6,436,359 B・SHA-256 `f8aab49b…`）と 4 分割は
  同フォルダに配置済み
- 未追跡 **27 件**（既存 24 ＋ 本フェーズの文書 3）。追跡・整理・削除はしていない
- U5 = FAIL / 既定 OFF、U3/U4 = HOLD、+50 µs と p99 基準は不変
- 製品コード・テスト変更なし、build / test / debugger / replay 未実行

## 10. 実装承認に必要な残件

1. `attachDiagSink` の呼び出し位置（`Init` か `prepareToPlay` か）と冪等性
2. DSPCore 群の生成タイミングと attach 順序
3. 診断 ON の ASan 構成（V4）を CI に追加するか（CI 定義の変更承認が必要）

**上記が確定すれば B-2 の実装承認に進めると判断する。** 現時点では HOLD を維持している。
