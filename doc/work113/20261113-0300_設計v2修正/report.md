# shutdown 設計 v2-a 修正（READ-ONLY）

- 作業日時: 2026-11-13 03:00 JST
- 基準 commit: `173e6995`（HEAD = origin/main）
- 性質: **設計のみ**。production source / test / CMake / git を変更していない。build / test / debugger / replay も、
  本稿のために実行していない（計測は前作業の範囲）。
- 入力:
  - 前稿 v2: `doc/work113/20261113-0200_設計改訂/report.md`
  - 計測: `doc/work113/20261113-0120_停止計測/report.md`
  - Owner 判定（2026-11-13）: A 要修正、B 方向妥当（再入・reconfigure 再確認）、C 契約具体化、D 妥当、
    E-5 要具体化、F は W1 の停止確認が不十分、実装 GO 保留、U-3 未承認。

---

## 0. 本稿の範囲と判定

本稿は v2 の **A（§3）・C（§5）・E-5（§7.2）・F の W1（§8.2）を置き換える**。
B は §5 で再確認のみ行い、設計は変えない。D は変更なし。

```
対象        = v2-a（A / B / C / E-5 / W1 の修正）だけ
対象外      = U-3（W3・W4・W7 の requestStopOnly）の実装設計
実装 GO     = 付与しない
Gate 1 承認 = 付与しない
```

Owner 指摘 4 点への対応は次のとおり。すべてソースで裏づけた。

| # | 指摘 | 対応 | 節 |
| --- | --- | --- | --- |
| 1 | A: intent と閉鎖完了の混同 | 3 状態（Intent / Closing / Closed）の CAS 状態機械に分離。閉鎖処理は MT が 1 回だけ実行し、producer は Intent の時点で拒否する | §1 |
| 2 | F: W1 の false positive | `isRunning == false` を停止の証明に使わない。ループ退出を専用フラグで publish し、起動前・起動中・退出済・join 済を区別する | §2 |
| 3 | C: epoch 抑止の適用範囲 | 9 箇所を個別に分類（抑止 / 許容）。MT 直列化による安全性の議論と、所有権移転の処理を確定 | §3 |
| 4 | E-5: 同期区間と lifecycle | `SyncWorkToken` の意味論、`beginTerminalStopAcceptingWork` からの検査、追加時のレビュー条件 | §4 |
| 5 | B: 再入・reconfigure の再確認 | JUCE の呼出し順を含めた再入シナリオ 4 件を列挙し、stuck 分岐が効く位置を確定 | §5 |

---

## 1. A の修正 — intent と閉鎖処理の分離

### 1.1 前稿の欠陥（Owner 指摘 1 の確認）

前稿 §3.2 は `if (exchangeAtomic(terminalStopRequested_, true, release)) return;` で冪等化していた。
一方 §7.1 は非 MT からの要求が同じ atomic を先に publish するとしていた。この組み合わせで次が成立する。

```
[非MT] requestTerminalShutdownIntent()   → terminalStopRequested_ = true
[MT]   beginTerminalStopAcceptingWork()  → exchange が true を返す → 即 return
                                           closeAdmission / requestShutdown / epoch++ が一度も走らない
```

結果、**意図は立っているのに admission は Open のまま**になる。`tryAdmit` で admission を取る producer
（RuntimePublication / Recovery / Build）はそのまま通れてしまう。Owner の指摘は正しい。

### 1.2 修正後の状態機械

`AudioEngine` に `std::atomic<uint8_t> terminalStopState_` を 1 個置く。値は次の 4 つ。

| 値 | 名前 | 意味 | 設定者 |
| --- | --- | --- | --- |
| 0 | `Idle` | 何も始まっていない | 初期値 |
| 1 | `Intent` | 終了の意図が記録された（閉鎖処理はまだ） | 任意スレッド |
| 2 | `Closing` | MT が閉鎖処理を実行中（この状態に入った 1 スレッドだけが実行する） | MT |
| 3 | `Closed` | 閉鎖処理が完了（admission 閉・publication shutdown 要求済・epoch 進展済） | MT |

遷移は CAS のみ。**後戻りしない。**

```
requestTerminalShutdownIntent()            // 任意スレッド・noexcept
    loop:
      s = load(acquire)
      if (s != Idle) return                // Intent / Closing / Closed は何もしない
      if (CAS(Idle → Intent, acq_rel)) return

beginTerminalStopAcceptingWork()           // Message Thread 専用・noexcept
    loop:
      s = load(acquire)
      if (s == Closed) return              // 完了済み
      if (s == Closing) return             // MT 上では起こらない（防御）
      if (CAS(s → Closing, acq_rel)) break // Idle からでも Intent からでも成立
    jassert(!syncWorkActive());                   // §4。MT 上で同期区間が開いていれば契約違反
    setShutdownPhase(ShutdownPhase::StopAcceptingWork, "beginTerminalStopAcceptingWork");
    shutdownRuntime_.closeAdmission();            // 冪等（ISRShutdown.cpp:453-454）
    runtimePublicationBridge_.requestShutdown();
    fetchAddAtomic(shutdownEpoch_, 1, acq_rel);   // §3 の callAsync 抑止に使う
    store(Closed, release)
```

**閉鎖処理の実行主体は「`Closing` への CAS に成功した 1 スレッド」だけ**である。intent の有無に関係なく、
MT が `beginTerminalStopAcceptingWork()` を呼べば必ず 1 回は実行される。前稿の欠陥はここで閉じる。

### 1.3 競合の全列挙

| 並行実行 | 結果 |
| --- | --- |
| 非 MT の intent が 2 本 | CAS は 1 本だけ成功。他方は `s != Idle` で return |
| intent → MT begin | MT の CAS(Intent→Closing) が成功し、閉鎖処理が走る（**前稿の欠陥はここ**） |
| MT begin → intent | intent の CAS(Idle→Intent) が失敗（state=Closing）。begin が閉鎖処理を実行 |
| MT begin → MT begin | 1 回目が Closed まで進む。2 回目は `s == Closed` で return |
| intent → MT begin → intent | 2 回目の intent は `s != Idle` で return |
| MT begin が途中で例外終了 | `noexcept` のため例外は出ない。`closeAdmission` 等はいずれも noexcept。もし将来 noexcept でない処理を足すなら、`Closing` のまま残ることを許容し、再入時に `Closing` を「実行中」とみなして return する（多重実行はしない） |

### 1.4 producer 側の述語

```
AudioEngine::isTerminalStopRequested() = terminalStopState_ >= Intent
AudioEngine::isStopAcceptingWork()     = terminalStopState_ >= Intent
                                      || consumeAtomic(shutdownPhase) >= StopAcceptingWork
```

**判定は `Intent` で既に真**にする。閉鎖処理の完了（`Closed`）を待たない。これにより
「intent は立ったが admission は Open」という窓の間も、§3.3 の明示的な入口（`loadIR` /
`loadImpulseResponse` / `startProgressiveUpgrade`）は新規作業を受け付けない。

`shutdownPhase` との OR を残すのは、`~AudioEngine` が `setShutdownPhase(StopAcceptingWork)` を
直接呼ぶ経路（`CtorDtor.cpp:160`）でも同じ判定が真になるようにするためである。

### 1.5 前稿から変わる点（差分）

| 項目 | 前稿 v2 | 本稿 |
| --- | --- | --- |
| 冪等化の単位 | `terminalStopRequested_` の exchange | `terminalStopState_` の CAS（Idle/Intent/Closing/Closed） |
| 非 MT の要求 | 同じ atomic を publish | `Intent` への遷移のみ |
| 閉鎖処理の実行 | exchange に勝った呼出し元 | `Closing` に CAS した MT |
| producer の拒否開始 | 閉鎖後（`StopAcceptingWork` 以降） | **intent の時点** |
| epoch の進展 | 前稿は閉鎖処理内 | 同じ（`Closing` → `Closed` の間で 1 回だけ） |

---

## 2. W1（`rebuildThread`）の停止確認 — false positive の排除

### 2.1 現状のフラグと実際の publish 位置（ソース実測）

| フラグ | 宣言 | 初期値 | 書き込み位置 |
| --- | --- | --- | --- |
| `rebuildThreadShouldExit` | `AudioEngine.h:3015` | false | `stopRebuildThread()`（`RebuildDispatch.cpp:820`）と起動時の clear（`PrepareToPlay.cpp:99`） |
| `rebuildThreadIsRunning` | `AudioEngine.h:3016` | false | ループ入口で true（`RebuildDispatch.cpp:874`）、**関数の最終行**で false（`:1449`） |
| `rebuildThreadStopped` | `AudioEngine.h:3021` | false | `join()` の**後**に true（`RebuildDispatch.cpp:832`）、起動時に false（`Init.cpp:41` / `PrepareToPlay.cpp:104`） |
| `rebuildThreadToken_` | `AudioEngine.h:3025` | 0 | ループ入口で thread id（`:878`）、退出時に 0（`:1451`） |

起動は 2 箇所（`Init.cpp:42`、`PrepareToPlay.cpp:109`）。どちらも `std::thread` の生成直後に戻るため、
**生成から `:874` に到達するまでの間、`IsRunning == false` かつ `Stopped == false`** という状態が実在する。
この窓で `IsRunning == false` を「停止済み」と解釈すると false positive になる。Owner の指摘は正しい。

`Stopped` は `join()` 後にしか立たない（`:828-832` のコメントも同旨）ので、**停止の唯一の確証**ではあるが、
preflight は join を行わない（join は無期限ブロックし得る）ため、そのままでは使えない。

### 2.2 状態の定義（5 状態）

| 状態 | 判定式（MT のみが評価） | 意味 |
| --- | --- | --- |
| `NotStarted` | `!rebuildThread.joinable()` | 未起動、または join 済み |
| `Starting` | `joinable && !IsRunning && !LoopExited` | 生成済みだがループ未到達（**前稿が見落とした窓**） |
| `Running` | `joinable && IsRunning && !LoopExited` | ループ実行中 |
| `Exited` | `joinable && LoopExited` | ループ関数を抜けた（join はまだ） |
| `Joined` | `!joinable && Stopped` | join 完了 |

「終了中」（ループを break してから関数末尾に達するまで）は `Exited` に入る前であり、`LoopExited == false` のままである。
待機側はこれを「未停止」として扱う（保守側）。`LoopExited` を break 直後に publish すると、残りの後始末
（`drainDeferredClearIfRequested` 等の後に置く場合）の間を「停止済み」と誤認するため、**publish は関数の最終行に固定する**。

### 2.3 追加する状態 `rebuildThreadLoopExited`

- 型: `std::atomic<bool>`（`AudioEngine.h` の W1 フラグ群に追加）。初期値 false。
- **設定者**: `rebuildThreadLoop()` のみ。`rebuildThreadIsRunning=false` と token=0 の**後**、
  関数の最終行で true を publish する（`RebuildDispatch.cpp:1449-1453` の直後）。
- **クリア**: スレッド生成の直前に、生成側（MT）が false を publish する。
  対象は `Init.cpp:41` と `PrepareToPlay.cpp:104` の 2 箇所（`rebuildThreadStopped=false` と同じ位置）。
- 不変条件: 「`LoopExited == true` は、そのスレッドがループ本体から抜けたことを意味する。
  `LoopExited == false` は、起動前・起動中・実行中のいずれかであり、停止とは解釈しない」。

### 2.4 preflight の判定（W1）

```
// MT のみ。W1 は「ループ退出」までを確認する。join はしない。
if (!rebuildThread.joinable()) return true;          // NotStarted / Joined
publishAtomic(rebuildThreadShouldExit, true, release);
rebuildCV.notify_all();                              // :823 と同じ起床
return waitUntil([this]{ return consumeAtomic(rebuildThreadLoopExited, acquire); }, deadline);
```

- `Starting` の窓でも `LoopExited == false` なので**停止済みとは判定しない**。
  この場合、ループは起床後に `ShouldExit` を見て即 break するため、期限內に `Exited` へ遷移する。
- `IsRunning` は判定に使わない（診断専用に降格する）。
- `Exited` を観測できた場合、その後の `join()` は直ちに戻る（ループ関数の残りは return のみ）。

### 2.5 preflight が証明しないこと（重要）

`LoopExited` は「ループ本体から抜けた」ことの証明であり、**スレッド終了（join 完了）の証明ではない**。
したがって:

1. 通常経路（S2）でも `stopRebuildThread()` の `join()` は必ず実行する（`:825-826`）。
   ただし preflight が `Exited` を確認済みなら、この join は待たない。
2. preflight の `bound` 超過で S3（保持）に入った場合、`join()` は呼ばない。
   `join` が無期限ブロックし得るためである（`RebuildDispatch.cpp:826`）。
3. 「preflight が成功した = W1 が join 可能な状態」という含意は成り立つが、
   「preflight が失敗した = W1 が動いている」は成り立たない（`Starting` で止まっている可能性がある）。
   失敗側の分岐は安全側（保持）に倒れるので、この非対称は許容する。

### 2.6 再起動との関係

`prepareToPlay()` は `!joinable()` のとき W1 を再生成する（`PrepareToPlay.cpp:95-111`）。
再生成時に `LoopExited` を false に戻さないと、**新しいスレッドが動き出す前に preflight が「退出済み」と誤認する**。
§2.3 のクリア位置はこのために必須である。

保持経路（S3）では `prepareToPlay` に到達しない（デバイスを開き直さない）ため、再起動との競合は起きない。
この前提は §5 の再入シナリオで確認する。

---

## 3. C — callback 抑止の適用範囲と競合契約

### 3.1 前提: Message Thread の直列性

`juce::MessageManager::callAsync` の本体は Message Thread で実行される。`beginTerminalStopAcceptingWork()` も
Message Thread 専用である。**Message Thread は協調的（非プリエンプティブ）なので、callback 本体の実行中に
shutdown の閉鎖処理が割り込むことはない。** 割り込みが生じるのは、callback 本体がメッセージループを回した場合だけであり、
それは §4 の E-1 で禁止している。

したがって「epoch を検査した直後に shutdown が始まる」競合は、**次の 2 通りに分解できる**。

| 競合 | 実際の順序 | 結果 |
| --- | --- | --- |
| 検査の直後に非 MT が intent を publish | intent は atomic。閉鎖処理はまだ走らない | callback は最後まで実行される。その後に MT が閉鎖する。**安全**（破棄は閉鎖後） |
| 検査の後、閉鎖処理より前に callback が終わる | MT 上で直列 | 同上 |
| 閉鎖処理が先、callback が後 | epoch 不一致 | callback は本体を実行しない（フィルタ） |
| 閉鎖処理が走らないままプロセス終了 | epoch は進まない | callback が走っても破棄は起きていないので安全 |

**この議論が成立する条件は「破棄（teardown）が MT でしか起きないこと」である。** v2-a の S3 では
teardown を一切実行しないので、条件は保たれる。

### 3.2 epoch の読み取り位置

```
enqueue 側（ワーカー / MT）:
    const uint64_t e = consumeAtomic(shutdownEpoch_, acquire);   // 所有権を release する前
    auto* raw = ptr.release();
    callAsync([e, raw]{ ... });
```

- epoch の読み取りは**所有権の移転より前**に行う。移転後に読むと、その間に閉鎖処理が走った場合に
  「新しい epoch を捕捉した callback」が生成され、フィルタをすり抜ける。
- 実行側は `consumeAtomic(shutdownEpoch_, acquire)` と比較する。不一致なら本体を実行しない。
- `shutdownEpoch_` は単調増加で、`Closing` に入った 1 スレッドだけが 1 回進める（§1.2）。

### 3.3 9 箇所の分類（対象 callback ごとの契約）

「本体」= 抑止対象の状態変更。「後始末」= 抑止しても必ず行う処理。

| # | 箇所 | 本体 | 所有権 | 判定 | 後始末 |
| --- | --- | --- | --- | --- | --- |
| C1 | `LoaderThread.cpp:87` | `isLoading` / `isRebuilding` の解除 | なし | **許容**（atomic のみ。止めると UI が固着する） | − |
| C2 | `LoaderThread.cpp:120` | `handleLoadError`（文字列） | なし（`juce::String` コピー） | **抑止**（破棄後の UI に触れない） | なし |
| C3 | `LoaderThread.cpp:288` | `finalizeNUCEngineOnMessageThread`（NUC 構築・IR 適用） | **4 本の生ポインタ**（`irLRaw` / `irRRaw` / `loadedIRRaw` / `displayIRRaw`）をラムダへ移転 | **抑止必須** | ラムダ先頭で RAII 化し、早期 return で解放（現行 `:304-307` の順序を維持） |
| C4 | `LoaderThread.cpp:331` | 失敗通知 | なし（`juce::String`） | **抑止** | なし |
| C5 | `ProgressiveUpgradeThread.cpp:218` | `applyComputedIR`（高解像度 IR の適用） | `preparedRaw`（`PreparedIRState`） | **抑止必須** | 現行どおり `unique_ptr` を先に生成（`:220`）。早期 return で解放済み |
| C6 | `LoadPipeline.cpp:843` | `executePendingCommit`（FINAL COMMIT） | `commitPtr`（`PendingCommit`） | **抑止必須** | `unique_ptr` を先に生成（`:846`）。早期 return で engine を解放 |
| C7 | `LoadPipeline.cpp:952` | `rebuildAllIRs()`（新しいロードの起動） | なし | **抑止**（§1.4 の入口拒否と二重の防御） | なし |
| C8 | `Rebuild.cpp:36` | `sendChangeMessage`（UI 通知） | なし | **許容**（購読者通知のみ。状態を変えない） | − |
| C9 | `StateAndUI.cpp:817` | レイテンシ表示更新 + `latencyChangePending` 解除 | なし | **許容**（解除を止めるとフラグが固着する） | − |

抑止対象は C2〜C7。C1・C8・C9 は「軽微な後始末」として許容する（Owner 指摘 3 の境界）。

### 3.4 所有権の移転が実行前に行われる箇所

C3・C5・C6 は `release()` してからラムダへ渡す。したがって**抑止した場合の解放責任はラムダ側**にある。
3 箇所とも現行コードがラムダ先頭で RAII 化しているので、判定は RAII 生成の**後**に置けばよい。

- C3: 判定は `:307` の後（4 本すべて RAII 化した後）。生ポインタを判定前に触らない。
- C5: 判定は `:220` の後。`PreparedIRState` は `unique_ptr` が破棄時にキャッシュ整合を保つ。
- C6: 判定は `:846` の後。`PendingCommit` のデストラクタが engine を解放する（`:857` のコメントと同旨）。

### 3.5 抑止しても実行しなければならない後始末

| 対象 | 理由 |
| --- | --- |
| C1 の `isLoading` / `isRebuilding` | 立てたままにすると UI と `timerCallback` の再入判定が固着する |
| C9 の `latencyChangePending` | 同上（次の表示更新が永久に抑制される） |
| C3/C5/C6 の RAII 解放 | メモリと engine の解放。抑止は「適用しない」であって「捨てない」ではない |

### 3.6 保持経路での挙動

S3（保持）では owner が生存しているため、`weakThis.get()` は非 nullptr を返す。
したがって**弱参照は抑止の役に立たない**（前稿の指摘どおり）。S3 では `terminalStopState_ == Closed` が
保たれ、epoch も進んだままなので、S3 中に積まれた callback はすべて epoch 不一致で本体を飛ばす。
`weakThis` は「万一 owner が破棄された場合」の最後の保険として残す（二重の防御）。

---

## 4. E-5 — 同期区間と shutdown lifecycle の整合

### 4.1 `SyncWorkToken` の意味論

単なる再入検出ではなく、**同期区間が lifecycle のどこに位置するかを表すトークン**にする。

```
class SyncWorkToken        // ConvolverProcessor のメンバ。非コピー
{
    // 生成時:
    //   jassert(!isStopAcceptingWork());      // S1 以降は同期作業を開始しない（E-3）
    //   jassert(!syncWorkActive_.exchange(true));   // ネスト禁止（E-5）
    // 破棄時:
    //   syncWorkActive_.store(false, release);
    //   publishAtomic(lastSyncWorkEndNs_, steadyNowNs(), release);   // テレメトリ
};
```

- `syncWorkActive_` は `std::atomic<bool>`。**MT 専用**（読み書きとも）。
- 対象区間: `loadIR()`、`loadImpulseResponse()`（同期実行部）、`rebuildAllIRsSynchronous()`、
  `applyComputedIR()`。将来追加される同期区間も対象（§4.4）。
- トークンは必ず関数の先頭で生成し、RAII で解除する（早期 return でも解除される）。

### 4.2 shutdown 側からの検査

`beginTerminalStopAcceptingWork()`（MT）は、`Closing` への CAS 直後・閉鎖処理の直前に `jassert(!syncWorkActive())` を置く（§1.2）。
MT 上で同期区間が開いている状態でこの関数が呼ばれるのは、**入れ子メッセージループで再入した場合だけ**であり、
それは E-1 違反である。Release では `jassert` が消えるため、次の 2 段で観測する。

1. `syncWorkActive` が true のまま閉鎖処理に入ったら、`diagLog("[DIAG] shutdown: sync work active at close")` を出す。
2. その場合でも閉鎖処理は実行する（中止すると admission が開いたままになる方が危険）。
   同期区間の側は、自身の終端で `isStopAcceptingWork()` を再確認し、適用（commit）を中止する。

### 4.3 状態遷移の整合（同期区間 × shutdown）

| 時点 | 同期区間 | shutdown 状態 | 整合の根拠 |
| --- | --- | --- | --- |
| 同期区間の開始 | 開始 | `Idle` のみ許可（`jassert`） | E-3。S1 以降は開始しない |
| 区間中に非 MT が intent | 継続 | `Intent` | teardown は MT なので走らない。区間は完走する（E-4） |
| 区間中に MT が閉鎖処理 | **起こらない**（E-1） | − | 入れ子ループ禁止 |
| 区間の終端 | 解除 | `Intent` の可能性 | 終端で `isStopAcceptingWork()` を再確認し、commit を中止できる |
| 区間の終端 → MT が閉鎖 | 解除済み | `Closed` | `jassert(!syncWorkActive)` を満たす |

### 4.4 契約の維持（レビュー条件）

E-1 は「現状そうである」だけでなく「今後も守る」契約である（前稿 §7.4 の全数確認:
production に `runDispatchLoop` / `ModalComponentManager` / `AlertWindow::show` は 0 件）。
維持のため、次のレビュー条件を課す。

| # | 条件 |
| --- | --- |
| R-1 | 同期区間を追加・変更する変更は、`SyncWorkToken` の生成位置と解除位置を明示する。 |
| R-2 | その区間の内側で `runDispatchLoop` 系 / モーダルループ / ファイルダイアログを呼ばないことをレビューで確認する。 |
| R-3 | `src/` の production に `runDispatchLoop` が 0 件であることを CI かレビューで機械確認する（grep 1 回で足りる）。 |
| R-4 | 同期区間の所要時間を計測できるようにする（`probe` の該当スパン、または同等のテレメトリ）。 |

R-3 は CI に置くと確実である。既存の ast-grep ゲート（`.github` 配下）に 1 ルール追加する形が最小。

### 4.5 `bound` との関係（再掲）

`bound` は Worker 待ち（W1・W2・W5・W6）にだけ適用する。同期区間には適用しない。
同期区間の所要は「shutdown 遅延の下限」として記録するが、上限は保証しない（計測 102.6〜144.0 ms / 334〜638 ms）。

---

## 5. B の再確認 — 再入と reconfigure 経路

Owner 指摘（B は方向妥当、実装前に再入・reconfigure を再確認）に対して、呼出し順を列挙する。

### 5.1 JUCE が `releaseResources()` を呼ぶ経路

前稿の調査（`juce_AudioProcessorPlayer.cpp`）どおり、次の 3 経路がある。

| # | 経路 | 呼ばれる pass | 本設計での扱い |
| --- | --- | --- | --- |
| B-1 | `setProcessor(nullptr)` → `oldOne->releaseResources()` | terminal（`terminalReleaseRequested_` を消費） | stuck なら入口で return（何も破棄しない） |
| B-2 | `setProcessor(p)` → 新 processor の `releaseResources()` → `prepareToPlay()` | 対象は新 processor。`AudioEngineProcessor` 側の実装次第 | 保持経路では `setProcessor(nullptr)` のみ呼ぶので B-2 は発生しない |
| B-3 | `audioDeviceStopped()` → `processor->releaseResources()` | terminal フラグは既に消費済み → **reconfigure pass** | stuck 判定で入口 return（learner 停止も飛ばす） |

B-3 が前稿で穴になり得た箇所である。`releaseResourcesForReconfigure()` は DSP/IRState を壊さない
（`ReleaseResources.cpp:811-833`）が、`noiseShaperLearner->stopLearning()` を含み、これが内部スレッドの join を
伴う（`NoiseShaperLearner.cpp:210-223`）。stuck 経路では待ちになるため、reconfigure pass の入口でも
`isWorkersStuck()` を見て return する。

### 5.2 `ConvolverProcessor::releaseResources()` の再入

`AudioEngineProcessor::releaseResources()` は `uiConvolverProcessor.releaseResources()` を呼ぶ
（`ReleaseResources.cpp:558` 付近）。stuck 経路では AudioEngine 側が入口で return するため、ここへは到達しない。
到達するのは通常経路だけであり、そのときは W5/W6 が停止済みなので `stopThread` は即時 true を返す。

### 5.3 再入の全列挙（stuck 経路）

| 順序 | 呼出し | 結果 |
| --- | --- | --- |
| 1 | `setProcessor(nullptr)` → terminal pass | 入口 return。`lifecycleState` は変わらない |
| 2 | `removeAudioCallback` | コールバック停止のみ |
| 3 | `closeAudioDevice` → `audioDeviceStopped` | B-3 の reconfigure pass → 入口 return |
| 4 | `stopTimer()` | タイマ停止 |
| 5 | 以降 | `~MainWindow` / `~AudioEngine` に到達しない（保持） |

`lifecycleState` が `Prepared` のまま残るが、保持経路では `prepareToPlay` も `~AudioEngine` も呼ばれないため
不整合にならない。`isStopAcceptingWork()` は `Closed` を返し続けるので、外部からの操作も拒否される。

### 5.4 残る注意点

- `releaseResources()` は `ASSERT_NON_RT_THREAD()`（`:36`）を持つ。B-3 はデバイススレッドから来る可能性があるが、
  この表明は既存の前提であり、本設計は変更しない（既存の挙動を変えない）。
- stuck 分岐は `isWorkersStuck()` の 1 読みのみで判断する。`terminalRelease` の値も併せて参照するため、
  分岐の判定順は「terminal フラグの消費 → stuck 判定 → 実行 or return」とする。

---

## 6. 不変条件（v2-a 版）

| # | 不変条件 |
| --- | --- |
| I-A1 | `terminalStopState_` は Idle → Intent → Closing → Closed の単調遷移のみ。 |
| I-A2 | 閉鎖処理（`closeAdmission` / `requestShutdown` / epoch 進展）は、`Closing` へ CAS した 1 スレッドだけが 1 回実行する。 |
| I-A3 | producer の拒否は `Intent` で始まる（閉鎖完了を待たない）。 |
| I-A4 | `requestWorkerStop()` は要求のみ。`stopThread` / `join` を呼ばない。 |
| I-B1 | `workersStuck_` が true の間、`AudioEngine::releaseResources()` は terminal / reconfigure の両 pass で何も破棄しない。 |
| I-B2 | stuck 経路では W1・W3・W4・W7 の `join` を呼ばない。W2・W5・W6 の `stopThread` を呼ばない。 |
| I-C1 | 抑止対象（C2〜C7）は epoch 不一致で本体を実行しない。軽微な後始末（C1・C9）は実行する。 |
| I-C2 | 抑止しても、移転済みの所有権は必ず解放する（RAII を判定より先に生成する）。 |
| I-C3 | callback 本体はメッセージループを回さない（E-1）。 |
| I-D1 | `workersStuck_` を false に戻す API を作らない。 |
| I-E1 | 同期区間の内側でメッセージループを回さない。 |
| I-E2 | 同期区間に `bound` を適用しない。 |
| I-E3 | `SyncWorkToken` はネストしない（`syncWorkActive_` の exchange で検出）。 |
| I-E4 | S1 以降は同期区間を開始しない（`jassert` + 入口拒否）。 |
| I-F1 | W1 の停止確認に `rebuildThreadIsRunning == false` を使わない。`rebuildThreadLoopExited` または `!joinable()` のみを使う。 |
| I-F2 | `rebuildThreadLoopExited` はループだけが true にし、生成側だけが false にする。 |
| I-F3 | 保持経路では `~MainWindow` / `~AudioEngine` / `~ConvolverProcessor` を実行しない。 |
| I-G1 | `~ConvolverProcessor` から retire provider を呼ばない（既存方針を維持）。 |

---

## 7. 実装単位（v2-a の内訳・見積り）

| # | 単位 | 内容 | 主な変更ファイル | 依存 |
| --- | --- | --- | --- | --- |
| P1 | 状態機械 | `terminalStopState_` / `shutdownEpoch_` / 2 API / 述語 | `AudioEngine.h` | − |
| P2 | W1 状態 | `rebuildThreadLoopExited` の追加と 3 箇所（ループ末尾・2 起動点） | `AudioEngine.h`, `RebuildDispatch.cpp`, `Init.cpp`, `PrepareToPlay.cpp` | − |
| P3 | preflight | `requestWorkerStop()` / `waitForWorkersToExit()`（W1/W2/W5/W6） | `AudioEngine.*`, `ConvolverProcessor.*` | P1, P2 |
| P4 | producer 拒否 | `loadIR` / `loadImpulseResponse` / `startProgressiveUpgrade` / `prepareToPlay` | `LoadPipeline.cpp`, `Lifecycle.cpp` | P1 |
| P5 | 非破壊分岐 | `releaseResources` の terminal / reconfigure 入口 | `ReleaseResources.cpp`, `ConvolverProcessor.Lifecycle.cpp` | P1 |
| P6 | callback 抑止 | C2〜C7 の 6 箇所 | `LoaderThread.cpp`, `ProgressiveUpgradeThread.cpp`, `LoadPipeline.cpp` | P1 |
| P7 | 同期区間 | `SyncWorkToken` と 4 区間への適用 | `ConvolverProcessor.*` | P1 |
| P8 | 保持 | `StuckProcessHolder` と audio-only teardown | `MainApplication.cpp`, `MainWindow.*` | P3, P5 |

P1 → P2 → (P3, P4, P5, P6, P7) → P8 の順で積める。P3 と P5 は独立に検証できる。

---

## 8. 未解決（v2-a の範囲で残るもの）

| # | 項目 | 内容 | 状態 |
| --- | --- | --- | --- |
| U-1 | `bound` | 5 秒は `kGracefulDrainMaxMs` の踏襲。W1・W2 は既存ログ（時刻付与が前提）、W5・W6 は未計測 | 未検証の暫定値 |
| U-2 | `stopLearning()` の所要 | stuck 経路では呼ばないが、通常経路の reconfigure で join する | 未計測 |
| U-4 | `Closing` のまま停止した場合 | 例外・強制終了で `Closed` に到達しないケース。再入時は `Closing` を「実行中」として return する設計（§1.3 最終行） | 設計済み・実装で固定 |
| U-5 | `shutdownAudioOnlyKeepingState()` の副作用 | `DeviceSettings::saveSettings` を呼ばない、ChangeListener 解除順序、`stopTimer` の位置 | 要設計（P8） |
| U-6 | stuck テレメトリ | 保持件数、`TimedOut` 理由、生存 Worker ID、`syncWorkActive`、`terminalStopState` | 要設計（P3/P8） |
| U-8 | 通常経路の `stopRebuildThread` join | preflight が `LoopExited` を確認済みでも join を実行する。join が戻らない異常（ループ退出後の後始末で停止）は観測できない | 受容（preflight 成功時のみ到達） |
| U-3 | W3・W4・W7 の停止要求 API | **本稿の対象外**（Owner 未承認） | 未承認 |

---

## 9. 証跡（本稿で追加確認したソース）

| 対象 | 位置 | 確認内容 |
| --- | --- | --- |
| W1 フラグ宣言 | `AudioEngine.h:3015-3025` | 3 フラグ + token の初期値はすべて false / 0。`Stopped` は join 後のみ true と明記 |
| W1 起動 | `Init.cpp:41-42` / `PrepareToPlay.cpp:104-109` | 生成前に `Stopped` を false にしている（`LoopExited` のクリアも同じ位置に置く） |
| W1 ループ入口 | `RebuildDispatch.cpp:874-879` | `IsRunning=true` と token publish はループ本体の前 |
| W1 ループ末尾 | `RebuildDispatch.cpp:1449-1453` | `IsRunning=false` → token=0 が関数の最終処理 |
| W1 停止 | `RebuildDispatch.cpp:815-832` | exit 要求 publish → notify → **無期限 join** → `Stopped=true` |
| W1 再起動 | `PrepareToPlay.cpp:95-111` | `!joinable()` のときだけ再生成 |
| admission | `ISRShutdown.cpp:445-470` | `closeAdmission()` は Open 以外で即 break（冪等・一方向） |
| 述語 | `AudioEngine.h:1626-1637` | `isShutdownInProgress()` は `shutdownRuntime_.phase_` 依存。admission 閉鎖では真にならない |
| C 対象 | `LoaderThread.cpp:87,120,288,331` / `ProgressiveUpgradeThread.cpp:218` / `LoadPipeline.cpp:843,952` / `Rebuild.cpp:36` / `StateAndUI.cpp:817` | 本体・所有権・RAII 位置を個別確認（§3.3） |
| B 再入 | `ReleaseResources.cpp:34-58, 236-238, 811-833` | terminal / reconfigure の分岐位置と、reconfigure が learner を止めること |

---

## 10. 停止条件の確認

- production source / test / CMake を本稿のために変更していない。
- build / test / debugger / replay を本稿のために実行していない。
- git の変更操作を行っていない。HEAD は `173e6995`。
- AGENTS.md の既存未コミット変更に触れていない。設計差分 v2.9 も不変。
- U-3 の実装設計には着手していない。

**本稿の提出をもって停止する。** 実装 GO・Gate 1 承認は付与しない。U-3 も未承認のままとする。
