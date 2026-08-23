#include <cstdio>

// Reproduce the Entry struct from TerminalReclaimAuthority (ISRRetireRouter.h:64-71)
// with identical field types and order, for sizeof verification.
enum class DeletionEntryType : unsigned char {
    Generic = 0,
    World   = 1
};

struct TestEntry {
    void* ptr = nullptr;
    void (*deleter)(void*) = nullptr;
    unsigned long long epoch = 0;
    DeletionEntryType type = DeletionEntryType::Generic;
    const char* reason = nullptr;
};

int main() {
    printf("sizeof(TestEntry) = %zu bytes\n", sizeof(TestEntry));
    printf("offsetof ptr:     %zu\n", __builtin_offsetof(TestEntry, ptr));
    printf("offsetof deleter: %zu\n", __builtin_offsetof(TestEntry, deleter));
    printf("offsetof epoch:   %zu\n", __builtin_offsetof(TestEntry, epoch));
    printf("offsetof type:    %zu\n", __builtin_offsetof(TestEntry, type));
    printf("offsetof reason:  %zu\n", __builtin_offsetof(TestEntry, reason));

    // Verify alignment
    printf("alignof(TestEntry) = %zu\n", alignof(TestEntry));
    printf("alignof(ptr)       = %zu\n", alignof(void*));
    printf("alignof(epoch)     = %zu\n", alignof(unsigned long long));
    printf("alignof(reason)    = %zu\n", alignof(const char*));

    if (sizeof(TestEntry) == 40) {
        printf("PASS: sizeof(Entry) == 40 bytes\n");
        return 0;
    } else {
        printf("FAIL: sizeof(Entry) != 40 (got %zu)\n", sizeof(TestEntry));
        return 1;
    }
}
