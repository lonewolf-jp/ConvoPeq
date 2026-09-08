# D135-8/9 Gate G-4.4 — Phase-I Durable Admission Audit（Work Report）

**Status: CONDITIONAL**（read-only 監査。Production source changes: 0 / Test source changes: 0）
**詳細:** `evidence/D135-8-9_GATE_G_G4-4_DURABLE_ADMISSION_AUDIT.md`
**ビルド evidence:** `evidence/g44_ctest.log`（本監査で取得: Debug full build→CTest `100% tests passed out of 40` DBG_CTEST_EXIT=0 / Release 同 `100% tests passed out of 40` REL_CTEST_EXIT=0。G-4.3-T-R の教訓により grep だけでない実測を併記）

## 中心命題 → 証明
「logical obligation の identity/ownership は admission table が保持し、durable slot は delivery representation に過ぎない」はコード上で成立:
- `tryInsert`（唯一の +1）は cpp:946 の 1 箇所のみ。durable 書込（submit cpp:998-1008 / redrive cpp:1158-1170）は既存 oblId の写像のみで生成しない。`liveCount_` 直接触达 0。
- coalesce（findByKey+state CAS、cpp:926-933）は table のみ参照、durable slot を読まない。durable 経路は CoalesceIdentity を再構築しない（層分離 D11 成立）。

## 必須監査項目の結論
| 項目 | 判定 | 要点 |
|---|---|---|
| D1 | ✓ | durable は obligation を生成しない（上） |
| D2 | ✓ | blind overwrite ガード cpp:991-996 は **obligationId ベース**の distinctness 保護。B は剥奪されず deferred 維持→毎 tick redrive。redrive 側は busy なら oblId 照合なしに transport へ退避（保守的） |
| D3 | ✓ | 同一 oblId 上書きのみ許可。buildSource 更新は D18.8 と整合（同一 {handle,target} の範囲で snapshot-level 最新化、canonical identity 不変）。recoveryGeneration 再生成なし。reservationOwned 冪等 |
| D4 | ✓ | 3 台帳分離（liveCount_ / pendingIntentCount_ / durable 占有）。delivery enum 排他・単一書込者。terminal 取り残しなし（durable 表現は一度消費で消える、resolve 冪等・ABA 安全） |
| D5 | ✓ | lease 完全追跡: take はクリアせず Building 化、failure→settle(true)→DurablePending、success/discard のみ最終解放。retry は liveCount_ 不変。spin 上限 4 で次サイクル委譲 |
| D6 | ✓ | Producer=CoordinatorLoop 単独（submit+redrive 同一スレッド）、Consumer=RebuildThread 単独（take/settle/rearm）。submit 経路の rebuildMutex+notify エッジが同一スレッドの redrive 書込にも happens-before を及ぼす。注記: take() は predicate を acquire 読みしない（実害なし、文書化推奨） |
| D7 | ✓ | resurrection 経路なし（redrive は Live のみスキャン cpp:1114/1140）。terminal 化済み durable 表現の再 build→resolve no-op（二重 −1 なし）。shutdown は discard+resolve の独立閉鎖 |
| D8 | ✓ | 期待構造と一致。C11-C16 全登録・両構成 PASS。差分は「durable busy（同一 oblId）でも transport 試行」— 観察 2 の一時的二重住処窓に直結（未検証領域） |
| D9 | ✓ | Q_max=256（h:927）/ L_residency_max=257（I4:943/984 = 256+1 の合成）/ L_logical_max=32（h:363）の三層混同なし。durable=1 が L を制限する誤推論はコード上成立しない（table 独立・park+redrive） |
| D10 | **Safety ✓ / Liveness 残存** | 最大残存リスク確定: **redrive 経路に Builder wake が配線されていない**（Threading.cpp:270 は recoveryPending/notify なし、wake 配線は submit 経路のみ AudioEngine.h:4500-4504）。Builder アイドル時に redrive 付着表現が次の無関係 wake まで滞留し得る（復旧遅延・安全側劣化、喪失なし）。dedicated retry timer なし。Coordinator tick 自体は 1ms fallback で保証 |
| D11 | ✓ | coalesce=admission / durable=residency の層分離をコード証明 |
| D12 | ✓ | EpisodeAdmissionState/GlobalRecoveryBudget/Tentative/SUPERSEDE/episode closure/per-episode capacity = src 全体 **0 ヒット**。RecoveryEpisodeId/ResolvedSuperseded/canSupersede の production 実装 0（G-4.3-T-R 既確認） |

## 残存事項（G-4.4 実装設計への入力。今回は一切変更しない）
1. **Liveness**: redrive 付着時の Builder wake ギャップ（D10-1）。
2. **一時的二重住処窓**: markTransientFailure の delivery=None 化（cpp:1078）が durable 保持中に発生→以後 transport 併存。有界の重複 build/publish のみ（台帳・世界整合は無傷）。C11-C16 未カバー。
3. **Building 中の上書き窓**: cpp:991 が sub-state 非確認。invalid payload discard（RECOVERY-6）との重畳で同一 oblId の fresh 表現が消え stranded 化し得る（極めて狭い。shutdown で回収）。
4. **コメントドリフト**: h:960-961（coalesce 判定用/coalesce で更新）— G-4.2 由来。

## 判定理由
- **CONDITIONAL**: Safety（blind overwrite / ownership loss / resurrection / 恒久的 double representation）は全て構造的に否定。一方 liveness 配線ギャップ・狭い二重住処/stranded 窓・コメントドリフトが限定残存。
- **NO-GO 不成立**: distinct Live obligation の delivery 喪失経路なし、所有保存の欠陥なし（CTest 40/40×2 + C9/C11-C16 実証）。
- **PASS 不成立**: D10 の liveness 残存と観察 2/3 が「coverage / evidence / compile boundary」ではなく設計残存問題として明示的に存在するため。

## STOP
G-4.4 Audit 完了・verdict 報告済み。**G-4.4 implementation / durable-table rewrite / retry redesign には進まない**（PASS/CONDITIONAL 判定时に別途指示との約束）。次の指示まで待機。
