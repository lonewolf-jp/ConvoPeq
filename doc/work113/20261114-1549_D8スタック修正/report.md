# D8 Stack Problem — 根本原因の特定と修正

- 実施日時: 2026-11-14 15:30〜15:50 JST（同日、再検証を追記）
- 対象: `D8_2_B_2_Tests`（Debug 構成）
- 種別: テストコードの修正（Production 変更なし・コミットなし）
- 基準 HEAD: `6e099932bcb4f122a72526f1c9786eecd7fc08b6`

---

## 1. 発生条件

| 項目 | 内容 |
| --- | --- |
| 実行ファイル | `D8_2_B_2_Tests.exe` |
| 構成 | **Debug のみ**（Release は変更前から PASS） |
| 失敗するテスト | `T13 C2 Non-RT only (success/Q/E/T/RT-violation/double-call)` |
| 症状 | 終了コード `0xC00000FD`（STATUS_STACK_OVERFLOW） |
| 発生箇所 | `test_T13_C2NonRtOnly()` の**関数入口**（フレーム確保時）。T13 の最初の `printf` より前に落ちる |
| 再現率 | 4/4（初回 2/2 ＋ 再検証 2/2。フレームサイズで決まるため決定的） |
| 発生スレッド | メインスレッド（T13 内の `std::thread` 生成より前） |
| 先行テスト | T1〜T12 はすべて PASS（`[T12] ... PASS` まで出力を確認） |

再現手順（実測）:

```
build\Debug\D8_2_B_2_Tests.exe
→ 終了コード 3221225725 (0xC00000FD)
```

クラッシュ位置の特定は、一時 worktree のコピーに `setvbuf(stdout, nullptr, _IONBF, 0)` を
入れて行った（本体のソースは変更していない）。出力は `[T12] ... PASS` で途切れ、`[T13]` の行が
出ないことから、T13 のフレーム確保時に落ちていると判定した。

---

## 2. 根本原因

`test_T13_C2NonRtOnly()` が **約 245 KiB の `convo::EpochDomain`（および同サイズの派生
`FullQueueEpochProvider`）を 5 個、スタック上に確保**していた。Debug 構成（`/Od`）はスコープごとに
別のスタック枠を割り当てるため、5 個分がそのまま 1 つのフレームに積み上がる。

| 実測値 | サイズ |
| --- | ---: |
| `sizeof(convo::EpochDomain)` | 251,328 B |
| `sizeof(FullQueueEpochProvider)` | 251,328 B |
| `sizeof(convo::isr::ISRRetireRouter)` | 66,192 B |
| `D8_2_B_2_Tests.exe` の stack reserve（Debug / Release 共通） | 1,048,576 B（`0x100000`） |

関数先頭のフレーム確保命令の即値を読むと、修正前の `test_T13_C2NonRtOnly` は次のとおりだった
（関数の同定方法は §9.2）。

```
mov eax,1955F8h
call @ILT+79080(__chkstk)
sub rsp,rax            ; 0x1955F8 = 1,660,408 バイト
```

この 1,660,408 B のフレームを 1,048,576 B のスタックに確保しようとして失敗していた。
内訳は `EpochDomain` 系 5 個（5 × 251,328 = 1,256,640 B）＋ `ISRRetireRouter` 5 個
（5 × 66,192 = 330,960 B）＋ その他のローカルと `/RTC1` の初期化領域である。

同じファイルの他テストも同様の構造だが、フレームが 1 MB 未満なので落ちない。

| 関数 | 修正前フレーム | 判定 |
| --- | ---: | --- |
| `test_T11_CompleteFadeRecords`（EpochDomain 2 個） | 668,488 B | 1 MB 未満 → PASS |
| `test_T12_FinalizeRetiresTargetAndCurrent`（1 個） | 332,168 B | 1 MB 未満 → PASS |
| `test_T13_C2NonRtOnly`（5 個） | **1,660,408 B** | **1 MB 超過 → 0xC00000FD** |

---

## 3. 問題となったオブジェクト・関数・呼び出し経路

- 関数: `test_T13_C2NonRtOnly()`（`src/tests/D8_2_B_2_Tests.cpp`）
- オブジェクト: `convo::EpochDomain`（`DeferredDeletionQueue` を内包。`kQueueSize = 4096` の
  ring buffer と sequence 配列で 246,016 B）と、その派生 `FullQueueEpochProvider`
- 呼び出し経路: `main()` → `test_T13_C2NonRtOnly()` のフレーム確保 → 失敗
- スタック上にあった宣言（修正前、5 箇所）:
  - `convo::EpochDomain dom(...)` × 2（scope 1 と scope 5）
  - `FullQueueEpochProvider fdom;` × 3（scope 2〜4）
- 同じパターンが T11（2 個）と T12（1 個）にもある（こちらは 1 MB 未満で PASS）

---

## 4. 修正内容と理由

**方針**: 巨大オブジェクトをヒープへ移す。スタックサイズ（`/STACK`）は変更しない。

各宣言を、寿命を変えずにヒープ確保＋参照エイリアスへ置き換えた（8 箇所）。

```cpp
// 修正前
convo::EpochDomain dom(convo::RetireDomainId::Unknown);

// 修正後
auto domStorage = std::make_unique<convo::EpochDomain>(convo::RetireDomainId::Unknown);
auto& dom = *domStorage;
```

`FullQueueEpochProvider fdom;` も同様に `std::make_unique<FullQueueEpochProvider>()` ＋ `auto& fdom`。

**この形を選んだ理由**:

| 要件 | 対応 |
| --- | --- |
| 寿命が従来と同等 | `unique_ptr` はスコープ終了時に破棄。宣言順が `domStorage` → `router` → `coord` なので、破棄順は `coord` → `router` → `dom` となり従来と同じ |
| 既存コードの変更を最小化 | 変数名 `dom` / `fdom` を参照のまま維持するため、以降の使用箇所（数十行）は無変更 |
| テストの意味を変えない | オブジェクトの型・初期化引数・利用 API は不変 |
| Production へ影響しない | `src/tests/` のみの変更。Production のヘッダ・実装は不変 |

スタックサイズを増やす方法は、原因（1 フレーム 1.66 MB）を残したまま閾値を上げるだけなので採用していない。

---

## 5. 変更ファイル

| ファイル | 変更 |
| --- | --- |
| `src/tests/D8_2_B_2_Tests.cpp` | 8 箇所の `EpochDomain` / `FullQueueEpochProvider` をヒープ確保へ変更（+28 / −8） |

`AGENTS.md` / `ConvoPeq.md` の既存の未コミット変更には触れていない（別ファイルのまま分離）。

---

## 6. スタック使用量の確認結果

関数先頭のフレーム確保命令（`mov eax,imm` ＋ `__chkstk`）の即値を読み、修正前後を比較した。
読み取りは `dumpbin /disasm` と PE `.pdata` の双方で行い、修正前の値は修正前ビルドの PDB
（`S_FRAMEPROC size`）とも一致する（§9.2）。

| 関数 | 修正前 | 修正後 | 削減 |
| --- | ---: | ---: | ---: |
| `test_T13_C2NonRtOnly` | 1,660,408 B | **403,568 B** | −1,256,840 B |
| `test_T11_CompleteFadeRecords` | 668,488 B | 165,728 B | −502,760 B |
| `test_T12_FinalizeRetiresTargetAndCurrent` | 332,168 B | 80,736 B | −251,432 B |

- 削減量は `EpochDomain` 1 個分（251,328 B）のほぼ整数倍である（T13 は 5 個分＋200 B、T11 は 2 個分＋104 B、T12 は 1 個分＋104 B。端数はアラインメントとその他のローカルの差）。
- 修正後 T13 のフレーム 403,568 B は、1 MB のスタックに対して **約 645 KB の余裕**がある。
- 残る 403 KB の主因は `ISRRetireRouter` 5 個（5 × 66,192 = 330,960 B）。
- **stack reserve は `0x100000`（1,048,576 B）のまま**。修正後の Debug と Release、修正前の Debug で
  `dumpbin /headers` により確認した。`/STACK` の指定は追加していない。

### 6.1 追加調査（再帰・コピー・構成差）

| 調査項目 | 結果 |
| --- | --- |
| 再帰呼び出し | **なし**。`D8_2_B_2_Tests.cpp` の全関数（`main` を含む 18 個）を本体の波括弧対応で切り出し、自己呼び出しを検索して 0 件 |
| 深い呼び出し | `main` → 各 `test_T*` の 1 段のみ。深いネストは無い |
| コピー・一時オブジェクト | なし。`convo::EpochDomain` は `std::atomic` と `DeferredDeletionQueue`（同様に atomic を内包）を持つため**非コピー**。テストは参照渡しのみ |
| 固定長キューの重複 | `EpochDomain` は 1 個につき 4096 エントリのキュー（246,016 B）を内包する。T13 は 5 個の provider を持つため、キューも 5 個分（設計どおり。テストが独立した provider を要求するため） |
| テスト fixture | 巨大メンバを保持する fixture は無く、各テスト関数がローカルに provider を直接構築していた（これが原因） |
| コンパイラ / 構成差 | 下表のとおり Debug と Release で挙動が分かれる |

**Debug / Release の差異**

| 項目 | Debug | Release |
| --- | --- | --- |
| 最適化 | `/Od` + `/RTC1` | `/O2`（+ LTCG） |
| スタック枠の割り当て | スコープごとに別枠 → 合算される | 再利用・インライン化で縮小 |
| `test_T13_C2NonRtOnly` のフレーム | 1,660,408 → **403,568 B**（実測） | 関数シンボルが無く（デバッグ情報なし）関数単位の実測は不可 |
| 実行結果 | 修正前 0xC00000FD → 修正後 PASS | 修正前から PASS（本修正は Release のコード生成を変えない） |
| stack reserve | `0x100000`（1,048,576 B） | `0x100000`（同） |

Release は 1 MB の reserve で PASS しており、フレームが 1 MB 未満であることは実行結果から確認できる。
関数単位の数値はシンボル不在のため取得できない（推測では埋めない）。

---

## 7. 実行したテストと結果

| # | テスト | 構成 | 結果 |
| --- | --- | --- | --- |
| 1 | `D8_2_B_2_Tests`（修正前） | Debug | **0xC00000FD**（T13 入口。§9.5 で 2/2 を直接再現） |
| 2 | `D8_2_B_2_Tests`（修正後・3 回） | Debug | **PASS 3/3**（T1〜T13 すべて PASS） |
| 3 | `D8_1_WrapperCacheTests` | Debug | PASS |
| 4 | `DomainTagTests` | Debug | PASS |
| 5 | `DeferredDeletionQueueReclaimTests` | Debug | PASS |
| 6 | `RetireGraceSemantics` | Debug | PASS |
| 7 | `ShutdownRetireIntentDrain` | Debug | PASS |
| 8 | `StuckReaderFallbackDrain` | Debug | PASS |
| 9 | `STG11EQRetire` | Debug | PASS |
| 10 | `ScopedEpochReader` / `ScopedEpochReaderStructural` | Debug | PASS / PASS |
| 11 | `D8_2_B_2_Tests` / `D8_1_WrapperCacheTests` | Release | PASS / PASS |

関連 10 テスト（#3〜#10 の ctest 実行）は **10/10 PASS**。

検証用の一時 worktree（`C:\VSC_Project\d8probe\ConvoPeq`、`6e099932`）でビルド・実行し、
確認後に撤去した。本体 worktree の未コミット変更（他タスク 38 ファイル）は触れていない。

**本体 worktree での再確認**（他タスクの未コミット変更 38 ファイルを含む現行ツリー）:
`cmake --build build --config Debug --target D8_2_B_2_Tests` で再ビルドし、
`D8_2_B_2_Tests.exe` を 3 回実行して **PASS 3/3**（exit 0）。stack reserve は `0x100000` のまま。
検証したソースは、一時 worktree で計測したものと SHA-256 が一致する
（`b55768fe7085f131...`）。

---

## 8. 未解決事項・残存リスク

| # | 内容 | 状態 |
| --- | --- | --- |
| 1 | Debug の `TailLayerIntegrityTests`（RES-F の CPU 予算超過） | 既知の別問題。本修正の対象外（D8 とは無関係） |
| 2 | 他のテストにもスタック上の `EpochDomain` がある（`STG10ReaderQuarantineTests`、`StuckReaderFallbackDrainTests`、`ScopedEpochReaderTests`、`V11ReaderReacquireTests`、`invariant_INV3_INV5`、`STG11EQRetireTests`） | いずれも 1 関数あたり 1〜2 個で 1 MB 未満のため PASS。**3 個以上に増えると同じ問題が再発する**（1 個 251,328 B） |
| 3 | 修正後 T13 のフレームは 403,568 B（`ISRRetireRouter` 5 個が主因） | 1 MB に対して余裕あり。ただしスコープを増やすと再び逼迫する |
| 4 | 全 ctest（Release） | **実施済み**（§9.6）。61 件中 60 件 PASS、失敗は `TailLayerIntegrityTests` 1 件のみ。単独実行では 2/2 PASS（CPU 予算 10 ms/s に対し 10.024 ms/s の負荷依存） |
| 5 | 全 ctest（Debug） | **実施済み**（§9.6）。61 件中 60 件 PASS、失敗は `TailLayerIntegrityTests` 1 件のみ（既知）。`D8_2_B_2_Tests` は PASS |

### 再発防止の目安

`EpochDomain`（251,328 B）をスタックに置く場合、**1 関数あたり 3 個以上**で 1 MB を超える。
Debug の `/Od` はスコープごとに枠を確保するため、「別スコープなら安全」とは限らない。
テストで複数の provider が必要な場合は、今回と同じく `std::make_unique` ＋参照で受ける形を標準とする。

---

## 9. 再検証（2026-11-14 追加）

作業ツリーの状態が変わっていないことと、数値が現行バイナリでも成立することを確認した。

### 9.1 前提状態

| 項目 | 実測値 |
| --- | --- |
| HEAD / origin/main | `6e099932bcb4f122a72526f1c9786eecd7fc08b6`（両者一致） |
| `src/tests/D8_2_B_2_Tests.cpp` の SHA-256 | `B55768FE7085F1312C3952F3AE0D229B0ECD3BD11E661EC98FE67C2518AE29CB`（検証済みビルドと一致） |
| 当該ファイルの差分 | +28 / −8（宣言 8 箇所のみ。検証条件・アサーションの削除なし） |
| staged / stash | 0 / 0 |

### 9.2 フレームサイズの再測定（シンボル非依存の方法）

`dumpbin /disasm` はこの exe の COFF シンボルが無いため関数名を出さない。そこで PE の
`.pdata`（RUNTIME_FUNCTION）から関数開始 RVA を列挙し、関数先頭 32 バイトから
`mov eax, imm32` ＋ 直後の `call`（`__chkstk`）の並びを探して、その即値をフレームとした。
MSVC は 4 KB 超のフレームでこの形を出すため、PDB やシンボルが無くても関数単位で読める。
関数の同定は、各テスト本文にしか現れない `printf` 文字列を `lea reg,[rip+disp32]` で
参照している命令を .text から探し、その RVA を含む `.pdata` エントリを取る方法によった。

現行 `build\Debug\D8_2_B_2_Tests.exe` の実測:

| 関数 | フレーム確保命令 | フレーム |
| --- | --- | ---: |
| `test_T13_C2NonRtOnly` | `mov eax,62870h` | **403,568 B** |
| `test_T11_CompleteFadeRecords` | `mov eax,28760h` | 165,728 B |
| `test_T12_FinalizeRetiresTargetAndCurrent` | `mov eax,13B60h` | 80,736 B |

exe 全体で 4 KB 超のフレームを持つ関数は 12 個で、最大は T13 の 403,568 B、次が 199,248 B。

修正前の値（1,660,408 / 668,488 / 332,168 B）は、修正前ビルドの PDB
（`build\Debug\D8_2_B_2_Tests.pdb`、2026-11-12 01:37）の `S_FRAMEPROC size` と一致する。
同 PDB の型情報は `sizeof(convo::EpochDomain) = 251,328`、
`sizeof(FullQueueEpochProvider) = 251,328`、`sizeof(convo::isr::ISRRetireRouter) = 66,192` を示す。

### 9.3 実行結果（再実行）

| 実行 | 結果 |
| --- | --- |
| `build\Debug\D8_2_B_2_Tests.exe` 単体 | exit 0。T1〜T13 のうち設計上の DEFERRED（T5 / T6 / T9）以外すべて PASS |
| `ctest -C Debug`（関連 11 件） | 11/11 PASS |
| `ctest -C Release`（D8 の 2 件） | 2/2 PASS |
| `dumpbin /headers` の stack reserve | `0x100000`（1,048,576 B、Debug） |

### 9.4 他テストのフレーム（再発余地の確認）

| 実行ファイル | 最大フレーム |
| --- | ---: |
| `D8_1_WrapperCacheTests` | 516,320 B（`EpochDomain` 2 個） |
| `D8_2_B_2_Tests`（修正後） | 403,568 B（T13） |
| `invariant_INV3_INV5Tests` | 319,880 B |
| `StuckReaderFallbackDrainTests` | 317,864 B |
| `ScopedEpochReaderTests` | 251,688 B（`EpochDomain` 1 個） |
| `DomainTagTests` | 4,672 B |

いずれも 1 MB 未満で、Debug 実行も PASS する。ただし 1 関数に 3 個以上置くと再発する点は
§8 のとおりで、変わらない。

### 9.5 修正前バイナリの直接再現（A/B）

修正前ソース（blob `09b82afae0ae7e24d4db4e5a03da48c59b3a6dcb`）でビルドされた
`C:\VSC_Project\ConvoPeq-ci-fix\build\Debug\D8_2_B_2_Tests.exe`（2026-11-12 09:48）が残っていた。
この blob は `25d68e9c` と `6e099932` で同一（`git rev-parse` で一致を確認）であり、当該 exe は
未修正ソースそのもののビルドに相当する。修正後は blob `11d5b79817f699ca3a5295ae887faa14f0d211da`。

| 実行 | 終了コード | 結果 |
| --- | --- | --- |
| 修正前 exe（2 回） | `0xC00000FD`（3,221,225,725 / −1,073,741,571） | T13 のフレーム確保で失敗。stdout はバッファリングのため空 |
| 修正後 exe（2 回） | 0 | T1〜T13 が PASS（T13 の行まで出力） |

修正前 exe のフレームは §9.2 と同じ方法で測り、1,660,408 B（rva `0x8E2D0`、code size 3537）、
T11 668,488 B、T12 332,168 B、stack reserve は `0x100000`。1 MB を超えるフレームを持つ関数は
T13 だけなので、失敗はこのフレーム確保に帰着する。

### 9.6 全 ctest（回帰確認）

作業ツリーの `CMakeLists.txt` には他タスクの未コミット変更（+143 行）が含まれ、テスト登録は
HEAD の 57 件ではなく **61 件**になっている。以下はその 61 件での結果。

| 構成 | 結果 | 所要 | 失敗したテスト |
| --- | --- | ---: | --- |
| Debug | 60/61 PASS | 224.9 s | `TailLayerIntegrityTests`（RES-F の CPU 予算超過。21.9〜31.3 ms/s、予算 10 ms/s） |
| Release | 60/61 PASS | 155.4 s | `TailLayerIntegrityTests`（10.024 ms/s。単独実行では 2/2 PASS） |

`D8_2_B_2_Tests` は Debug（`5/61 ... Passed 0.16 sec`）と Release（2/2）の両方で PASS した。
`TailLayerIntegrityTests` は `src/tests/TailLayerIntegrityTests.cpp` 単独のターゲットで、D8 の
テスト TU をリンクしない。修正前から記録されている既知の失敗（RES-F の CPU 予算）であり、
本修正の影響ではない。
