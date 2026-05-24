# CodeQL 即対応リスト（Critical / High）

作成日: 2026-05-24
対象SARIF: `storage/codeql/query-runs/convopeq-cpp-run-20260524-3/20260524-104321/results.sarif`

## 優先順位ルール

- **P0**: Critical（即時トリアージ）
- **P1**: High かつ **自リポ編集可能領域（`src/`）**
- **P2**: High のうち件数が多い外部依存（再発/波及が大きい）
- **P3**: High の単発外部依存（アップストリーム追従 or baseline化候補）

> 注: 本リポ規約により `JUCE/` と `r8brain-free-src/` は原則編集禁止。外部依存の検出は「自リポ修正」ではなく「アップストリーム更新/差分管理/除外設定」で扱う。

## 自リポで直せる項目（今回）

- **修正対象（P1 / High / 1件）**: `src/eqprocessor/EQProcessor.ProcessingCache.cpp`（L48 / `cpp/incorrect-allocation-error-handling`）
- **修正方針**:
   1. L48付近のアロケーション失敗判定を、実際のアロケータ仕様（nullを返す/返さない）と照合する。
   2. 仕様上不要なnullチェックは削除し、必要な失敗処理のみを残す。
   3. 失敗ハンドリングの表現をプロジェクト方針に統一する（例外 or 明示エラー処理）。
- **完了条件**: 修正後に one-step CodeQL を再実行し、`cpp/incorrect-allocation-error-handling` が 0 件であることを確認する。
- **実施結果（2026-05-24）**: `src/eqprocessor/EQProcessor.ProcessingCache.cpp` の該当検出は **0件**（再解析SARIF: `storage/codeql/query-runs/convopeq-cpp-run-20260524-fix1/20260524-115139/results.sarif`）。

---

## P0（Critical）

1. **JUCE/modules/juce_graphics/fonts/harfbuzz/hb-blob.cc**
   - 件数: Critical x1
   - ルール: `cpp/type-confusion`
   - 代表位置: L145
   - 対応方針: **JUCE更新で吸収可否を最優先確認**。固定版運用なら一時的にCodeQL baseline化（抑制理由を記録）。

2. **JUCE/modules/juce_graphics/fonts/harfbuzz/hb-ot-cmap-table.hh**
   - 件数: Critical x1
   - ルール: `cpp/unsigned-difference-expression-compared-zero`
   - 代表位置: L1645
   - 対応方針: 上記と同様（アップストリーム確認優先）。

---

## P1（High / 自リポ即修正）

1. **src/eqprocessor/EQProcessor.ProcessingCache.cpp**
   - 件数: High x1
   - ルール: `cpp/incorrect-allocation-error-handling`
   - 位置: L48
   - 対応方針（即修正）:
     - null戻りを想定した分岐が実際のアロケータ仕様と不整合か確認
     - 不要なnullチェックを除去、または失敗ハンドリングを例外/契約に合わせて統一
     - 修正後に同一ワンステップで再解析し、当該ルール消失を確認

---

## P2（High / 件数多の外部依存）

1. **JUCE/modules/juce_audio_formats/codecs/oggvorbis/libvorbis-1.3.7/lib/psy.c**
   - 件数: High x5
   - 主ルール: `cpp/integer-multiplication-cast-to-long`

2. **JUCE/modules/juce_audio_formats/codecs/oggvorbis/libvorbis-1.3.7/lib/vorbisfile.c**
   - 件数: High x3
   - 主ルール: `cpp/alloca-in-loop`

3. **JUCE/modules/juce_graphics/geometry/juce_EdgeTable.cpp**
   - 件数: High x3
   - 主ルール: `cpp/suspicious-pointer-scaling` / `cpp/alloca-in-loop`

4. **JUCE/modules/juce_audio_processors_headless/processors/juce_AudioProcessorGraph.cpp**
   - 件数: High x2
   - 主ルール: `cpp/inconsistent-null-check`

5. **JUCE/modules/juce_graphics/fonts/harfbuzz/hb-ot-shaper-arabic-fallback.hh**
   - 件数: High x2
   - 主ルール: `cpp/integer-multiplication-cast-to-long`

6. **JUCE/modules/juce_graphics/fonts/harfbuzz/hb-ot-var-gvar-table.hh**
   - 件数: High x2
   - 主ルール: `cpp/integer-multiplication-cast-to-long`

7. **JUCE/modules/juce_graphics/image_formats/jpglib/jquant2.c**
   - 件数: High x2
   - 主ルール: `cpp/integer-multiplication-cast-to-long`

対応方針:

- JUCEの上位バージョン差分確認（同一箇所修正済みか）
- 修正未取り込みなら、依存更新計画に束ねる（個別パッチは原則回避）
- 直近運用は baseline（既知外部依存）として管理

---

## P3（High / 単発の外部依存）

以下は各1件（外部依存）:

- `JUCE/modules/juce_audio_basics/mpe/juce_MPEInstrument.cpp` (`cpp/integer-multiplication-cast-to-long`)
- `JUCE/modules/juce_audio_devices/native/juce_ASIO_windows.cpp` (`cpp/integer-multiplication-cast-to-long`)
- `JUCE/modules/juce_audio_formats/codecs/flac/libFLAC/lpc_flac.c` (`cpp/integer-multiplication-cast-to-long`)
- `JUCE/modules/juce_audio_formats/codecs/oggvorbis/libvorbis-1.3.7/lib/res0.c` (`cpp/integer-multiplication-cast-to-long`)
- `JUCE/modules/juce_audio_formats/codecs/oggvorbis/libvorbis-1.3.7/lib/sharedbook.c` (`cpp/integer-multiplication-cast-to-long`)
- `JUCE/modules/juce_audio_utils/gui/juce_AudioThumbnail.cpp` (`cpp/integer-multiplication-cast-to-long`)
- `JUCE/modules/juce_core/json/juce_JSON.cpp` (`cpp/inconsistent-null-check`)
- `JUCE/modules/juce_core/memory/juce_HeapBlock.h` (`cpp/uncontrolled-allocation-size`)
- `JUCE/modules/juce_core/native/juce_Network_windows.cpp` (`cpp/incorrect-string-type-conversion`)
- `JUCE/modules/juce_core/native/juce_SystemStats_windows.cpp` (`cpp/suspicious-add-sizeof`)
- `JUCE/modules/juce_core/zip/zlib/zutil.c` (`cpp/integer-multiplication-cast-to-long`)
- `JUCE/modules/juce_events/interprocess/juce_InterprocessConnection.cpp` (`cpp/unsafe-use-of-this`)
- `JUCE/modules/juce_graphics/fonts/harfbuzz/hb-directwrite.cc` (`cpp/incorrect-allocation-error-handling`)
- `JUCE/modules/juce_graphics/fonts/harfbuzz/hb-machinery.hh` (`cpp/suspicious-pointer-scaling`)
- `JUCE/modules/juce_graphics/fonts/harfbuzz/hb-open-type.hh` (`cpp/integer-multiplication-cast-to-long`)
- `JUCE/modules/juce_graphics/fonts/harfbuzz/hb-ot-kern-table.hh` (`cpp/integer-multiplication-cast-to-long`)
- `JUCE/modules/juce_graphics/fonts/harfbuzz/hb-ot-layout-common.hh` (`cpp/integer-multiplication-cast-to-long`)
- `JUCE/modules/juce_graphics/fonts/harfbuzz/OT/glyf/glyf-helpers.hh` (`cpp/integer-multiplication-cast-to-long`)
- `JUCE/modules/juce_graphics/image_formats/jpglib/jctrans.c` (`cpp/suspicious-pointer-scaling`)
- `JUCE/modules/juce_graphics/image_formats/jpglib/jdcoefct.c` (`cpp/suspicious-pointer-scaling`)
- `JUCE/modules/juce_graphics/image_formats/jpglib/jdmainct.c` (`cpp/integer-multiplication-cast-to-long`)
- `JUCE/modules/juce_gui_basics/widgets/juce_Slider.cpp` (`cpp/integer-multiplication-cast-to-long`)
- `JUCE/modules/juce_gui_basics/widgets/juce_TableListBox.cpp` (`cpp/inconsistent-null-check`)
- `JUCE/modules/juce_gui_basics/widgets/juce_TreeView.cpp` (`cpp/inconsistent-null-check`)
- `JUCE/modules/juce_gui_extra/code_editor/juce_CodeEditorComponent.cpp` (`cpp/integer-multiplication-cast-to-long`)

---

## 直近アクション（実行順）

1. `P1` の `src/eqprocessor/EQProcessor.ProcessingCache.cpp` を修正
2. 再度 one-step CodeQL を実行し、P1消込みを確認
3. `P0/P2/P3` は外部依存管理チケットへ移管（JUCE更新計画 or baseline管理）
