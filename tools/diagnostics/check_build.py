import os, glob

src = "C:/VSC_Project/ConvoPeq/src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp"
src_mtime = os.path.getmtime(src)
print("Source mtime:", src_mtime)

for f in glob.glob("C:/VSC_Project/ConvoPeq/build/**/*DSPCore*Double*", recursive=True):
    if os.path.isfile(f):
        obj_mtime = os.path.getmtime(f)
        print("obj:", f)
        print("obj mtime:", obj_mtime)
        print("newer:", obj_mtime > src_mtime)

# Also search for the specific source in the build tree
for f in glob.glob("C:/VSC_Project/ConvoPeq/build/**/AudioEngine.Processing.DSPCoreDouble*", recursive=True):
    if os.path.isfile(f):
        print("found:", f)
        print("size:", os.path.getsize(f))
