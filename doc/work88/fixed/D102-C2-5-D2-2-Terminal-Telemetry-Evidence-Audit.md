# D102-C2-5-D2-2 — Terminal Telemetry Evidence Audit

- **実施日**: 2026-08-26
- **作業種別**: **read-only / audit-only**（production source 変更 **0** / test 変更 **0** / CMake 変更 **0** / contract 変更 **0**）
- **基準版**: ConvoPeq.md `Generated: 2026-08-26 20:06:47`
- **前提**: D2-1 NO-GO（Phase A）— bounded Terminal 不採用、telemetry 起動条件付き保留

---

## E1. telemetry semantics — source 確定 … **PASS**

`TerminalReclaimAuthority`（`ISRRetireRouter.cpp:27-118`, `ISRRetireRouter.h:92-119`）:

| Counter | 更新点 | 実装 |
|---|---|---|
| `terminalStoreCount_` | `store()` 成功時（L38） | no-op（ptr/deleter null, L30-31）は push_back 前に return → **カウントされない**（一貫） |
| `residentAtomic_` | store ++(release) / drain −pending / drainAll =0 | lock-free 現在値 |
| `terminalPeakResident_` | store 内 CAS ループ（L40-48, residentAtomic を authoritative source に使用） | lifetime high watermark、**減算経路なし** |
| `terminalDrainAllCount_` | `drainAll()` 冒頭（L95）のみ | epoch-gated `drain()` では**増加しない** |
| `terminalDrainEntryCount_` | `drainAll()` の実行エントリごと（L108）のみ | 同上 |
| `reclaimCount_` | World 型 deleter 実行時（L85, L112） | drain / drainAll 両方 |

### カウンタ意味の成立確認（store → resident++ → peak 更新 → drain → resident--）

```text
store()   : push_back + residentAtomic_++ + storeCount_++ + peak CAS更新   … ✅ 成立
drain()   : epoch-safe 抽出 → deleter 実行 → residentAtomic_ -= n          … ✅ 成立
            ※ cumulative counter は不変（E2 の重要な性質、下記参照）
drainAll(): 全強制解放 + residentAtomic_=0 + drainAllCount++ + drainEntryCount += n … ✅ 成立
```

`residentCountAtomic()`（Router 合算, `ISRRetireRouter.h:349-352`）= Q + EmergencyQ + Terminal。
CoordinatorLoop の event-driven wake predicate（`ISRRetireRouter.cpp:487-489`）はこの合算を使用 —
**Terminal も drain predicate 対象になっていることを source で確認**。

**特記事項（E2 への重要な含意）**: epoch-gated `drain()`（L52-90）は
`residentAtomic_` のみを減算し、`terminalDrainEntryCount_` を増加させない。
→ **`storeCount − drainEntryCount ≠ resident` が成立する**（通常の差分推定が使えない）。

---

## E2. cumulative / current 分離 … **PASS**

| telemetry | 種類 | sizing 使用可 | 備考 |
|---|---|---|---|
| `terminalResident`（snapshot 値） | **現在値** | ○ | drain 後 0 へ復帰、authoritative |
| `terminalPeakResident` | **lifetime peak** | ○（単独では限定的） | 減算されないため transient spike と sustained growth を**単独では区別不能** |
| `terminalStoreCount` | 累積 store | ○ | admission レート推定に使用可 |
| `terminalDrainEntryCount` | 累積 drain 件数 | △ | **drainAll 分のみ**。epoch-gated drain 分は未計上（E1 特記） |
| `terminalDrainAllCount` | 累積 drainAll 回数 | ○ | shutdown/強制解放の発生頻度指標 |

### 滞留判定能力の検証

```text
必要な判定: 「peak ↑」が一時的スパイクか持続的増加か

可能な組み合わせ:
  peak ↑ + resident → 0 遷移観測          → 一時的滞留（transient）と判定できる
  peak ↑ + resident が非ゼロで持続        → 滞留継続の疑い
  storeCount ↑ 速度 vs drain 観測         → throughput 収支の概算

限界:
  - 単一 snapshot の peak/resident では区別不可 → **時系列 capture が必要**（E5 分類 B）
  - storeCount − drainEntryCount による収支計算は不可（drain() 未計上のため）
    → residentCountAtomic を一次情報に使うこと
```

**値そのものからの「leak / 持続的増加」断定は構造的に不可能** — 判定は
resident の時系列遷移観測に依存する。これは telemetry 設計の欠陥ではなく、
sizing 判断に時系列 capture が必要という要件を意味する。

---

## E3. observation path audit … **PASS**

追跡結果（`AudioEngine.h:1656-1705` `getRuntimeBackpressureTelemetry()`）:

```text
ISRRetireRouter terminal accessors
      ↓ (atomic load / mutex 保護付き count)
getRuntimeBackpressureTelemetry()
      ├─ terminalReclaimResidentCount()   … L1663-1664  ← Terminal resident 取得 ✅
      ├─ terminalStoreCount()             … L1694       ← ✅
      ├─ terminalDrainAllCount()          … L1695       ← ✅
      ├─ terminalDrainEntryCount()        … L1696       ← ✅
      ├─ terminalPeakResident()           … L1697       ← ✅
      ├─ activeReaderCount / minReaderEpoch / pendingRetireCount /
      │   emergencyQuarantineResidentCount … L1666-1673  ← Q/E/D との同時関連付け ✅
      ↓
RuntimeBackpressureTelemetry struct（L1588-1600 フィールド定義、「for K_terminal sizing」コメント明記）
      ↓
消費者: AudioEngine.Timer.cpp:1213-1217 / 1603-1604（診断ログ出力）
        AudioEngine.Threading.cpp:152（collectDrainAudit、shutdown 完了条件）
```

| 確認項目 | 判定 |
|---|---|
| 1. snapshot が Terminal resident を取得 | ✅ L1663-1664 |
| 2. peak/store/drain counters 取得 | ✅ L1694-1697 |
| 3. destructive operation か | ❌ 非破壊 — 全 accessor は atomic load / const count（`residentCount()` は mtx_ lock の読取のみ） |
| 4. RT thread から unsafe API | ⚠️ **制約**: `terminalReclaimResidentCount()` は mutex を取るため RT 呼出し禁止。現行 caller は Timer（message thread）/ Threading collectDrainAudit（NonRT）のみで RT caller なし（rg 確認）。**将来 RT から呼ばないこと**という使用上の契約として記録 |
| 5. stale/shadow state か | ❌ live 値 — X6 §6.6 により aggregate 上書きは廃止済み、router から直接 atomic load |
| 6. Q/E/D と同一 snapshot 関連付け | ✅ 単一関数の単一 return で全フィールド構築（厳密には各 atomic load が逐次だが、correlated capture 契約としては十分） |

---

## E4. 既存 Terminal telemetry tests 監査 … **PASS**

`src/tests/TerminalTelemetryContractTests.cpp`（T-5.1〜T-5.6）:

| Test | 検証内容 | 確認 |
|---|---|---|
| T-5.1 initial state | 全 counter == 0 | ✅ L14 |
| T-5.2 after 3 stores | storeCount==3, peakResident==3, residentAtomic==3 | ✅ L35 |
| T-5.3 after drain | **resident → 0**, reclaimCount(World) 加算 | ✅ L60 |
| T-5.4 **peak monotonicity** | peak が単調非減少、store ごとに正確に増加 | ✅ L104-121（`peak < prevPeak` 不変条件） |
| T-5.5 drainAll | drainAllCount==1, drainEntryCount==4, **resident==0, storeCount 保持(4), peak 保持(4)**、再 drainAll で entryCount 不変 | ✅ L145-186 |
| T-5.6 Generic/World 分離 | type 別 reclaim 計上分離 | ✅ L188 |

→ 「peak は単調増加・drain 後も保持」「resident は drain で 0 復帰」まで実証済み。
**telemetry counter の正確性は確立済み**。

### 重要な分離（ユーザー指示通り）

> 「counter が正しく動く」（T-5.x で実証済み）≠「実運用で bounded Terminal が必要」
>
> 後者は **負荷下での時系列データ** が初めて判定可能になる。既存テストは unit-level の
> counter 正確性のみを担当し、ISRSoakTests は「publish を含まないデータ構造耐久」
> （CMakeLists.txt コメント明記）であり、**Terminal 滞留の時系列観測はどこにも存在しない**。

---

## E5. K-terminal 起動条件の証拠性分類

D2-1 起動条件: 「`terminalPeakResident` の持続的増加 ＋ memory pressure の health telemetry 裏付け」

| # | 証拠項目 | 分類 | 根拠 |
|---|---|---|---|
| 1 | Terminal 現在滞留（resident） | **A: 既に観測可能** | snapshot フィールド + Timer ログ出力済み |
| 2 | Terminal lifetime peak | **A** | 同上 |
| 3 | store/drainAll 発生頻度 | **A** | 同上（drainEntryCount は drainAll 限定の注意付き） |
| 4 | Q/E/D/reader/epoch との相関 | **A** | 単一 snapshot で同時取得（E3-6） |
| 5 | retire pressure level / escalation | **A** | snapshot 内（retirePressureLevel, retireEscalationCount）— health 裏付けの一次材料 |
| 6 | **peak の transient/sustained 区別** | **B: 時系列記録が必要** | 単発 peak では判別不能。soak 下での periodic capture＋解析が必須 |
| 7 | **負荷下での store/drain レート収支** | **B** | resident 時系列＋storeCount 差分（※drainEntryCount は使えない）から推定 |
| 8 | heap 使用量への直接的帰属 | **C: 現行 telemetry では判定不能** | エントリ数→バイト数変換テーブルなし。ただし K sizing は件数ベースで足りる（C は blocking ではない） |

結論: **起動条件の証拠は「A（即時）+ B（時系列 capture 要）」で充足可能。**
C 項目は sizing を block しない。

---

## E6. 最終判定

```text
E1 telemetry semantics ........ PASS
E2 cumulative/current split ... PASS
E3 observation path ........... PASS
E4 existing tests ............. PASS
E5 evidence sufficiency ....... PASS
E6 re-entry condition ......... PASS-B
```

### **PASS-B**

根拠:
1. telemetry 自体は種類・網羅性・取得経路とも **bounded 再審判断に十分**
   （D2-1 で要求した全 counter が存在し、単一 snapshot で相関取得でき、unit test で正確性実証済み）
2. ただし「持続的増加」判定は **soak/負荷下の時系列 capture データが前提**であり、
   現状は Timer ログ出力があるものの系統的な capture・解析パイプラインが存在しない
3. E1 特記事項（epoch-gated drain が drainEntryCount 未計上）を分析時に遵守すれば、
   新規 production instrumentation は不要

### production source change: **0** / test change: **0** / contract change: **0**

（本監査は read-only。禁止事項リストの全項目を実施せず）

---

## 次ステップ（PASS-B 遷移）

**「soak telemetry capture design」を実施し、実測値が得られるまで D2-1 Phase B/C は保留。**

設計時に確定すべき事項（design-only）:
1. ISRSoakTests への `RuntimeBackpressureTelemetry` periodic capture 追加の要否
   （test 変更となるため、別途実施指示を仰ぐ）
2. capture interval と判定窓（例: peak と resident の推移を N 秒間隔で記録）
3. 分析ルール: 「resident 非ゼロ持続 + peak 単調更新 + storeRate > drainRate」を
   sustained growth の操作定義とする
4. E1/E2 の注意を分析仕様に明記: drainEntryCount ベースの収支計算禁止、
   residentCountAtomic を一次情報に使用

---

## 附録: 分析時の注意（E1/E2 から導出）

```text
⚠️ 禁止: leak 推定に (terminalStoreCount − terminalDrainEntryCount) を使用すること
    → epoch-gated drain() 分が未計上のため過大評価になる

✅ 推奨: residentCountAtomic()（または snapshot.terminalResident）の時系列を primary、
         terminalPeakResident を upper-bound 証拠、
         terminalStoreCount を admission-rate proxy として使用
```
