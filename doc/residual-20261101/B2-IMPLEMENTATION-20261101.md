# B-2 実装作業記録（2026-11-01）

- ブランチ: `fix/b2-diag-sink-20261101`（`main` = `6bb7c8d5` から作成）
- 承認: Owner 指示（B-2 実装承認・build/test 実行許可・検証後の main 統合許可）
- 方式: スライス分割（各スライスでビルドを通し、原因の混同を避ける）

## スライス 1（実装済み・ビルド確認中）

### 1.1 `deferredDestroy_` の型付き化（Owner 指示 1「型付き化」）

| 箇所 | 変更前 | 変更後 |
|---|---|---|
| `AudioEngine.h`（宣言） | `void deferDestroyRetry(void* p)` | `void deferDestroyRetry(DSPCore* core)` |
| `AudioEngine.h`（メンバ） | `std::vector<void*> deferredDestroy_;` | `std::vector<DSPCore*> deferredDestroy_;` |
| `AudioEngine.Threading.cpp`（呼出） | `owner->deferDestroyRetry(p)` | `owner->deferDestroyRetry(static_cast<DSPCore*>(p))` |
| 同（実装） | 引数 `void* p`、`for (void* existing : ...)` | 引数 `DSPCore* core`、`for (DSPCore* existing : ...)` |
| 同（再試行） | `std::vector<void*> snapshot;` / `for (void* p : snapshot)` | `std::vector<DSPCore*> snapshot;` / `for (DSPCore* p : snapshot)` |

**入口の `void*` は残る**（`destroyDSPCoreNode(void*)` が `void(void*)` の関数ポインタとして
EBR キューへ渡されるため）。ただし DSPCore 破棄は「唯一の漏斗」であり、
キャストはその契約が文書化された入口 1 箇所に限られる。teardown の走査は型付きで行う。

### 1.2 `DiagSink` の導入（非所有バンドル）

`AudioEngine.h` に `struct DiagSink` を追加した。**バッファ本体の所有権は AudioEngine に残す。**

```cpp
struct DiagSink {
    using Buffer = LockFreeRingBuffer<DiagEvent, DiagRuntimeLimits::BufferCapacity>;
    Buffer* buffer = nullptr;
    DiagPerTickCounter* tickPushed = nullptr;
    DiagPerTickCounter* tickDropped = nullptr;
    std::atomic<uint64_t>* totalPushed = nullptr;

    bool valid() const noexcept;
    bool push(const DiagEvent&) const noexcept;   // 固定長リング + relaxed カウンタのみ
};
```

`push()` は確保・ロック・I/O を行わない（RT 安全性は従来経路と同一）。

### 1.3 `DSPCore` の ownerId と sink 管理

```cpp
std::uint64_t diagOwnerId_ { 0 };              // 0 = 未束縛（予約値）
std::atomic<DiagSink*> diagSink_ { nullptr };

void attachDiagSink(DiagSink* sink, std::uint64_t ownerId) noexcept;
void detachDiagSink() noexcept;
```

- `attachDiagSink` は **NonRT（prepare）からのみ**呼ぶ。
- **別エンジンの sink へは bind しない**（`diagOwnerId_` が既存かつ不一致なら拒否）。
  `assert` は不変条件の検出として残し、Release でも `if` で拒否する。
- `detachDiagSink` はポインタのみ外す。`ownerId` は識別情報なので保持する。
- `ownerId` には既存の **`engineInstanceId_`** を使う（プロセス内で一意・単調・0 予約・再利用なし）。
  新しい採番器は追加しない。

### 1.4 AudioEngine 側の受け渡し

```cpp
DiagSink diagSink_ {};                              // diagBuffer の隣
DiagSink& diagnosticsSink() noexcept;
std::uint64_t diagnosticsEngineId() const noexcept; // engineInstanceId_
```

`AudioEngine::Init` で `diagSink_` の参照先を設定（既存の `setEqDiagBuffer` / `setConvDiagBuffer` は
スライス 2 で撤去するまで併存）。`DSPCore::prepare` の `if (owner != nullptr)` 内で
`attachDiagSink(&owner->diagnosticsSink(), owner->diagnosticsEngineId())` を実行する
（publish より前・NonRT）。

### 1.5 teardown 時の detach

`~AudioEngine` の「回収できずに残した DSPCore」ブロックの直後に、型付き走査で detach を追加。

```cpp
{
    std::lock_guard<std::mutex> lock(deferredDestroyMutex_);
    for (DSPCore* core : deferredDestroy_)
        if (core != nullptr) core->detachDiagSink();
}
```

順序は `StopAudio`（RT join）→ Coordinator/Builder join → drain → retryDeferredDestroys →
**detach** → member teardown。RT は停止済み、reader は sink を使わないため安全。

## スライス 3（実装済み・テスト PASS）

### 3.1 追加したテスト

`src/tests/AudioEngineHarness/B2DiagSinkTests.cpp`（opt-in: `B2_DIAGSINK_TEST=1`）。
既定スイートには影響しない（`main` の環境変数分岐で早期 return）。

| 検査 | 内容 | 結果 |
|---|---|---|
| unbound sink | `valid()==false`、`push()` が false | PASS |
| bound sink | `push()` 成功、pushed/total カウンタ +1 | PASS |
| ring full | `push()` が false、dropped カウンタ +1 | PASS |
| fresh core | `diagSink()==nullptr`、`diagOwnerId()==0` | PASS |
| ownerId 0 | attach 拒否 | PASS |
| null sink | attach 拒否 | PASS |
| 初回 attach | 受理（sink と ownerId が一致） | PASS |
| 別 owner | attach 拒否（既存 bind を保持） | PASS |
| 同一 owner 再 attach | 冪等 | PASS |
| detach | sink は nullptr、ownerId は保持 | PASS |

実行結果: `[B2DS] RESULT failures=0`（全 14 項目 PASS）。

### 3.2 M1〜M7 の測定値（Release / 診断 OFF / MSVC 19.51）

| # | 項目 | 実測値 |
|---|---|---|
| M1 | `sizeof(AudioEngine::DSPCore)` | **4,624,576 B** |
| M2 | `diagOwnerId_` のオフセット（アドレス差） | **4,624,552** |
| M3 | `diagSink_` のオフセット（アドレス差） | **4,624,560** |
| M4 | `std::is_standard_layout_v<DSPCore>` | **false（0）** |
| M5 | `std::atomic<DiagSink*>::is_always_lock_free` / `is_lock_free()` | **true / true** |
| M6 | `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` | **0** |
| M7 | コンパイラ | **msvc 1951** |

**M4 = false が実測で確定した。** これまで「規格要件からは非標準レイアウトと導けるが実測未了」と
していた論点は、**実測で非標準レイアウトと確定**した。したがって `offsetof` を採用しない判断は
（実測に基づく）妥当であり、アドレス差による測定は標準レイアウト性に依存しない。

M1〜M7 は**記録のみ**で、合否判定や安全性判定には使っていない。

### 3.3 ビルド・テスト結果

| 構成 | ビルド | テスト |
|---|---|---|
| `build-ci-local` / Release / 診断 OFF | 成功 | `AudioEngineHarness` **PASS（298.60 / 298.90 s）** |
| `build-diag` / Debug / 診断 ON | 成功 | `AudioEngineHarness` 実行中（assert 有効） |
| B-2 専用テスト（Release / 診断 OFF） | — | **PASS（failures=0）** |
| B-2 専用テスト（Debug / 診断 ON） | — | **PASS（failures=0）** |

### 3.4 診断 ON 構成の測定値（Debug / 診断 ON / MSVC 19.51）

| # | 項目 | 実測値 |
|---|---|---|
| M1 | `sizeof(DSPCore)` | **4,657,600 B**（OFF は 4,624,576 B、差 **33,024 B**） |
| M2 | `diagOwnerId_` オフセット | 4,657,584 |
| M3 | `diagSink_` オフセット | 4,657,592 |
| M4 | `is_standard_layout_v` | **false**（両構成で同一） |
| M5 | `atomic<DiagSink*>` lock-free | **true / true**（両構成で同一） |
| M6 | 診断マクロ | 1 |
| M7 | コンパイラ | msvc 1951 |

**構成間でレイアウトは一致しない**（`#if` 内に非静的メンバが 3 個あるため）。
要求どおり「1 バイナリ内で単一のマクロ値」は保たれており、構成正ごとに値を記録した。

## スライス 2（実装済み・ビルド確認済み）

1. `AudioEngine.Processing.DSPCoreFloat.cpp`
   - モジュールレベル `eqDiagBuffer` / `eqTickPushed` / `eqTickDropped` / `eqTotalPushed` と
     `setEqDiagBuffer` を撤去。
   - `logEqTime(...)` に `DiagSink* sink` を追加し、`sink->push(event)` に統一（カウンタは
     `DiagSink::push` 内で更新）。
   - 呼び出し 2 箇所に `diagSink()` を渡す。
2. `AudioEngine.Processing.DSPCoreDouble.cpp`
   - 同 `logEqTime` を sink 引数化（呼び出し 2 箇所）。
   - ANS switch の push を `diagSink()` 経由へ。
3. `AudioEngine.Processing.DSPCoreIO.cpp`
   - ANS switch の push を `diagSink()` 経由へ。
4. `AudioEngine.h`
   - `setEqDiagBuffer` / `setConvDiagBuffer` の宣言と `eqDiagBuffer` 系 `extern` を撤去。
5. `AudioEngine.Init.cpp`
   - 旧 setter 呼び出しを撤去（`diagSink_` の参照先設定のみ残す）。
6. `ConvolverProcessor.h` / `ConvolverProcessor.Runtime.cpp`
   - `process(block)` → `process(block, const DiagSink* sink = nullptr)`。
   - モジュールレベル `convDiagBuffer` 系と `setConvDiagBuffer` を撤去。
   - CONV_TIME / STCONV_TIME の 2 push を `sink->push(event)` に変更（null なら push しない）。
   - 呼び出し 4 箇所（DSPCoreFloat 2、DSPCoreDouble 2）で `diagSink()` を渡す。

**RT 経路の性質は変えていない**: `DiagSink::push` は固定長リングへの push と relaxed カウンタ更新のみ。
確保・ロック・I/O・delete は行わない。

## スライス 3（未着手）

- B-2 専用テスト（同一 owner の bind、別 owner の拒否、detach 後の push 無効、二重 bind の冪等）。
- M1〜M7 の測定・記録。

## 検証

| 段階 | 状態 |
|---|---|
| スライス 1 ビルド（Release / 診断 OFF） | **成功** |
| スライス 2 ビルド（Release / 診断 OFF） | **成功** |
| スライス 2 ビルド（Debug / 診断 ON） | **成功**（既存 warning 1 件のみ） |
| B-2 専用テスト（Release） | **PASS（failures=0）** |
| 既定スイート（Release） | スライス 1 時点で **PASS（298.71 s）**。スライス 2 後に再実行中 |

### ビルド手順の注意（実測）

| 事象 | 原因 / 対処 |
|---|---|
| `build.bat Release nopause` が generator 不一致で失敗 | 既存 `build` は **Visual Studio 18 2026** で構成済み。`build.bat` は **Ninja Multi-Config** を要求する |
| `cmake --build build-ci-local ...` で `'cstdio'` / `'atomic'` が見つからない | `vcvars64.bat` を読まずに起動したため `INCLUDE` / `LIB` が未設定。**必ず vcvars64 経由で実行する** |
| `CMakeLists.txt` 変更後の再構成で MKL が見つからなくなる | `$ENV{MKLROOT}` / `$ENV{IPPROOT}` は**構成時に評価**される。ビルド前に必ず設定する |
| 使用するビルドツリー | `build-ci-local`（Ninja / Release / 診断 OFF）、`build-diag`（Ninja / 診断 ON）、`build-asan-local`（Ninja / RelWithDebInfo）。**いずれも削除・再構成しない** |

実行コマンド（実績）:

```bat
cd /d C:\VSC_Project\ConvoPeq
set "MKLROOT=C:\Program Files (x86)\Intel\oneAPI\mkl\latest"
set "IPPROOT=C:\Program Files (x86)\Intel\oneAPI\ipp\latest"
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
cmake --build build-ci-local --config Release --target AudioEngineHarness -j 8
```

B-2 専用テストの実行:

```bat
set B2_DIAGSINK_TEST=1
build-ci-local\Release\AudioEngineHarness.exe
```
