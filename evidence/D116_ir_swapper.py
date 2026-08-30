import sys, time, shutil, os
# Cyclically swap active IR file content: A -> B -> C -> A ...
srcs = [r"C:\VSC_Project\ConvoPeq\evidence\D116_irA.wav",
        r"C:\VSC_Project\ConvoPeq\evidence\D116_irB.wav",
        r"C:\VSC_Project\ConvoPeq\evidence\D116_irC.wav"]
active = sys.argv[1]
duration_s = float(sys.argv[2])
interval_s = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
tmp = active + ".swaptmp"
i = 0
deadline = time.time() + duration_s
while time.time() < deadline:
    shutil.copyfile(srcs[i % 3], tmp)
    os.replace(tmp, active)  # atomic on same volume
    print(f"swapped to IR-{chr(65 + i % 3)} at {time.time():.1f}", flush=True)
    i += 1
    time.sleep(interval_s)
print("swapper done", flush=True)
