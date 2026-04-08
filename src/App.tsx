import React from 'react';

function App() {
  return (
    <div className="min-h-screen bg-neutral-900 text-white flex items-center justify-center p-8">
      <div className="max-w-2xl w-full bg-neutral-800 rounded-2xl p-8 shadow-2xl border border-neutral-700">
        <h1 className="text-4xl font-bold mb-4 tracking-tight">ConvoPeq</h1>
        <p className="text-neutral-400 mb-6 leading-relaxed">
          A high-performance JUCE convolution equalizer project.
          The source files have been successfully organized in the <code>/src</code> directory.
        </p>
        <div className="space-y-4">
          <div className="p-4 bg-neutral-900 rounded-lg border border-neutral-700">
            <h2 className="text-sm font-semibold text-neutral-500 uppercase tracking-wider mb-2">Project Status</h2>
            <p className="text-green-400 font-mono text-sm">Source files: Ready</p>
            <p className="text-green-400 font-mono text-sm">JUCE Framework: Ready (v8.0.12)</p>
            <p className="text-green-400 font-mono text-sm">r8brain-free-src: Ready</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
