import { FormatTranscript } from "@/components/format";

export default function Home() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 sticky top-0 z-10 backdrop-blur-sm bg-white/90">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center gap-3">
          <div className="h-8 w-8 rounded-lg bg-indigo-500 flex items-center justify-center flex-shrink-0">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-slate-900">Transcript Formatter</h1>
            <p className="text-xs text-slate-500">AI-powered transcript formatting</p>
          </div>
        </div>
      </header>
      <main className="max-w-5xl mx-auto px-6 py-8">
        <FormatTranscript />
      </main>
      <footer className="border-t border-slate-200 bg-white/60 py-3 text-center">
        <span className="text-xs text-slate-400 font-mono">
          build {process.env.NEXT_PUBLIC_GIT_SHA}
        </span>
      </footer>
    </div>
  );
}
