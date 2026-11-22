// Measurement-only QPC marks for issue #236. Not committed.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <windows.h>

namespace m236
{
    struct Mark
    {
        const char* tag;
        long long qpc;
        long long cpu;
    };

    inline long long threadCpu100ns() noexcept
    {
        FILETIME creation, exit, kernel, user;
        if (GetThreadTimes (GetCurrentThread(), &creation, &exit, &kernel, &user) == 0)
            return 0;
        const long long k = (static_cast<long long> (kernel.dwHighDateTime) << 32) | kernel.dwLowDateTime;
        const long long u = (static_cast<long long> (user.dwHighDateTime) << 32) | user.dwLowDateTime;
        return k + u;
    }

    inline Mark* buffer() noexcept
    {
        static Mark marks[256];
        return marks;
    }

    inline int& count() noexcept
    {
        static int n = 0;
        return n;
    }

    inline void mark (const char* tag) noexcept
    {
        LARGE_INTEGER c;
        QueryPerformanceCounter (&c);
        int& n = count();
        if (n < 256)
        {
            buffer()[n].tag = tag;
            buffer()[n].qpc = c.QuadPart;
            buffer()[n].cpu = threadCpu100ns();
            ++n;
        }
    }

    inline void dump() noexcept
    {
        const char* path = std::getenv ("CONVOPEQ_M236_TRACE");
        if (path == nullptr)
            return;

        FILE* f = nullptr;
        if (fopen_s (&f, path, "w") != 0 || f == nullptr)
            return;

        for (int i = 0; i < count(); ++i)
            std::fprintf (f, "%s %lld %lld\n", buffer()[i].tag, buffer()[i].qpc, buffer()[i].cpu);

        std::fclose (f);
    }

    struct AutoDump
    {
        ~AutoDump() { dump(); }
    };
}
