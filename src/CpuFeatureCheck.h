//==============================================================================
// CpuFeatureCheck.h
// ★ [P0-1] AVX2 ランタイム検出 — 非対応 CPU ではエラーダイアログを表示して終了
//
// ISR 観点: 起動時に 1 回だけ呼ばれるチェック。ISR とは無関係。
//==============================================================================
#pragma once

namespace convo {

// AVX2 + FMA が利用可能かをチェックする。
// 非対応の場合は Windows MessageBox でエラーを表示し、false を返す。
// 対応している場合は true を返す。
bool checkAVX2SupportAndWarn() noexcept;

} // namespace convo
