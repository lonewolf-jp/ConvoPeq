// i3-4-K fixture: UTF-8 (BOM-less) source with Japanese identifiers, comments, string literals.
// Includes Shift-JIS hazard chars (表 ソ 十 構) whose CP932 encodings contain 0x5C.
#include <cstdio>
#include <cstring>

// 日本語コメント: この行が Shift-JIS として誤解釈されるとパースが崩れる可能性がある（表ソ十構）
static const char* g_text = "日本語テスト表ソ十構";

int main() {
    int 変数 = 42;  // UTF-8 identifier
    std::printf("IDENT_OK=%d\n", 変数);
    std::size_t n = std::strlen(g_text);
    std::printf("LEN=%zu\n", n);
    for (std::size_t i = 0; i < n; ++i) std::printf("%02X ", (unsigned char)g_text[i]);
    std::printf("\n");
    return 0;
}
