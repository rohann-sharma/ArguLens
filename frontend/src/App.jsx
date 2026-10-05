import { useState } from 'react'
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle,
  HelpCircle,
  XCircle,
  Search,
  Info,
  ChevronDown,
  ChevronUp,
  FileText,
  Activity,
  ShieldAlert,
  Link as LinkIcon
} from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts'

const samples = [
  "According to the CDC report in 2023, regular handwashing reduces the spread of respiratory illnesses by up to 21%. Therefore, public health campaigns should emphasize hand hygiene.",
  "According to a 2024 Harvard study of 20,000 teenagers, 80% of teenagers who use Instagram experience depression. Therefore Instagram should be banned for people under 18.",
  "The sky is blue today. Thus, climate change is a complete hoax and unrelated to human activity.",
  "All politicians are corrupt. John is a politician, so John must be corrupt. Anyone who disagrees is just as corrupt.",
  "Pineapple on pizza is an absolute crime. People who like it just have bad taste."
];

function VerificationCard({ result, evidence }) {
  const [expanded, setExpanded] = useState(false);

  const getStatusConfig = (status) => {
    switch (status) {
      case 'FOUND_SUPPORTED':
        return { icon: <CheckCircle className="text-green-500 w-5 h-5 mt-1 shrink-0" />, color: 'text-green-400', bg: 'bg-green-900/20' }
      case 'FOUND_PARTIALLY_SUPPORTED':
        return { icon: <AlertTriangle className="text-yellow-500 w-5 h-5 mt-1 shrink-0" />, color: 'text-yellow-400', bg: 'bg-yellow-900/20' }
      case 'FOUND_CONTRADICTED':
        return { icon: <XCircle className="text-red-500 w-5 h-5 mt-1 shrink-0" />, color: 'text-red-400', bg: 'bg-red-900/20' }
      case 'FOUND_BUT_UNCLEAR':
        return { icon: <HelpCircle className="text-gray-400 w-5 h-5 mt-1 shrink-0" />, color: 'text-gray-300', bg: 'bg-gray-800' }
      case 'NOT_FOUND':
        return { icon: <Search className="text-gray-500 w-5 h-5 mt-1 shrink-0" />, color: 'text-gray-400', bg: 'bg-gray-800' }
      case 'NO_SOURCE_PROVIDED':
        return { icon: <Info className="text-gray-600 w-5 h-5 mt-1 shrink-0" />, color: 'text-gray-500', bg: 'bg-gray-800/50' }
      default:
        return { icon: <AlertCircle className="text-gray-500 w-5 h-5 mt-1 shrink-0" />, color: 'text-gray-400', bg: 'bg-gray-800' }
    }
  }

  const config = getStatusConfig(result.status);
  const evText = evidence?.find(e => e.id === result.evidence_id)?.text || "Unknown evidence";

  return (
    <div className={`rounded-lg border border-gray-700 overflow-hidden ${config.bg}`}>
      <div 
        className="p-4 cursor-pointer hover:bg-gray-700/30 transition flex items-start gap-3"
        onClick={() => setExpanded(!expanded)}
      >
        {config.icon}
        <div className="flex-1">
          <div className="flex items-center justify-between">
            <h4 className={`font-semibold ${config.color}`}>{result.status.replace(/_/g, ' ')}</h4>
            {expanded ? <ChevronUp className="w-5 h-5 text-gray-400" /> : <ChevronDown className="w-5 h-5 text-gray-400" />}
          </div>
          <p className="text-sm text-gray-300 mt-1 italic">"{evText}"</p>
          <p className="text-sm text-gray-400 mt-2">{result.details}</p>
        </div>
      </div>
      
      {expanded && (
        <div className="p-4 border-t border-gray-700 bg-gray-900/50 space-y-4 text-sm">
          {result.best_match ? (
            <div className="space-y-2">
              <h5 className="font-semibold text-blue-300 flex items-center gap-2">
                <FileText className="w-4 h-4" /> Best Source Match
              </h5>
              <div className="bg-gray-800 p-3 rounded space-y-1 text-gray-300">
                <p><span className="text-gray-500">Title:</span> {result.best_match.title}</p>
                <p><span className="text-gray-500">Provider:</span> {result.best_match.provider} {result.best_match.year && `(${result.best_match.year})`}</p>
                {result.best_match.url && (
                  <p><a href={result.best_match.url} target="_blank" rel="noreferrer" className="text-blue-400 hover:underline flex items-center gap-1"><LinkIcon className="w-3 h-3"/> Source Link</a></p>
                )}
                {result.best_match.abstract_snippet && (
                  <p className="mt-2 text-xs text-gray-400 border-l-2 border-gray-600 pl-2">{result.best_match.abstract_snippet}...</p>
                )}
              </div>
            </div>
          ) : (
            <p className="text-gray-500 italic">No source metadata available.</p>
          )}

          <div className="grid grid-cols-2 gap-4">
            <div>
              <h5 className="font-semibold text-gray-400 mb-1">Metrics</h5>
              <ul className="space-y-1 text-gray-300">
                <li>Match Score: <span className="font-mono bg-gray-800 px-1 rounded">{result.source_match_score.toFixed(2)}</span></li>
                {result.evidence_comparison && (
                  <li>TF-IDF Sim: <span className="font-mono bg-gray-800 px-1 rounded">{result.evidence_comparison.similarity.toFixed(2)}</span></li>
                )}
              </ul>
            </div>
            {result.numerical_mismatches?.length > 0 && (
              <div>
                <h5 className="font-semibold text-red-400 mb-1">Numerical Mismatches</h5>
                <div className="flex gap-1 flex-wrap">
                  {result.numerical_mismatches.map((m, i) => (
                    <span key={i} className="bg-red-900/30 text-red-300 border border-red-800 px-2 py-0.5 rounded text-xs">
                      {m}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {result.searched_providers?.length > 0 && (
            <div>
              <h5 className="font-semibold text-gray-400 mb-1">Providers Searched</h5>
              <div className="flex gap-2 flex-wrap text-xs">
                {result.provider_results.map((pr, i) => (
                  <span key={i} className={`px-2 py-1 rounded ${pr.status.includes('unavailable') ? 'bg-red-900/30 text-red-400' : 'bg-gray-800 text-gray-300'}`}>
                    {pr.provider}: {pr.status}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleAnalyze = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    
    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      })
      
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`Failed to analyze: ${res.status} ${txt}`);
      }
      
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 md:p-8 font-sans selection:bg-blue-500/30">
      <div className="max-w-7xl mx-auto space-y-8">
        
        <header className="border-b border-gray-800 pb-6 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent inline-flex items-center gap-2">
              <Activity className="w-8 h-8 text-blue-500" />
              ArguLens
            </h1>
            <p className="text-gray-400 mt-1">AI-Powered Argument Quality & Evidence Verification</p>
          </div>
        </header>

        <main className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* LEFT COLUMN: Input */}
          <section className="lg:col-span-5 space-y-6">
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 shadow-lg space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold flex items-center gap-2 text-gray-200">
                  <FileText className="w-5 h-5 text-blue-400" /> Input Text
                </h2>
                <select 
                  className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-300 focus:ring-2 focus:ring-blue-500 outline-none max-w-[200px]"
                  onChange={(e) => setText(samples[e.target.value])}
                  defaultValue=""
                >
                  <option value="" disabled>Load sample...</option>
                  <option value="0">1. Medical Fact</option>
                  <option value="1">2. Unverifiable citation</option>
                  <option value="2">3. Unrelated evidence</option>
                  <option value="3">4. Logical fallacy</option>
                  <option value="4">5. No evidence</option>
                </select>
              </div>

              <textarea
                className="w-full h-64 bg-gray-950/50 border border-gray-700 rounded-lg p-4 text-gray-200 focus:ring-2 focus:ring-blue-500 outline-none resize-y placeholder:text-gray-600"
                placeholder="Paste an argument, article paragraph, social media post, or transcript here..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
              <button
                onClick={handleAnalyze}
                disabled={loading || !text.trim()}
                className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-gray-800 disabled:text-gray-500 disabled:cursor-not-allowed text-white font-semibold py-3.5 rounded-lg transition-all flex items-center justify-center gap-2 shadow-lg shadow-blue-900/20"
              >
                {loading ? (
                  <><Activity className="w-5 h-5 animate-pulse" /> Analyzing Pipeline...</>
                ) : (
                  <><Search className="w-5 h-5" /> Analyze Argument</>
                )}
              </button>
              {error && (
                <div className="p-3 bg-red-900/20 border border-red-800/50 rounded-lg text-red-400 text-sm flex items-start gap-2">
                  <AlertCircle className="w-5 h-5 shrink-0" />
                  <p>{error}</p>
                </div>
              )}
            </div>
          </section>

          {/* RIGHT COLUMN: Results */}
          <section className="lg:col-span-7 space-y-6">
            {!result && !loading && (
              <div className="h-full min-h-[400px] border border-gray-800 border-dashed rounded-xl flex items-center justify-center text-gray-500 bg-gray-900/30">
                <p>Enter text and analyze to see the breakdown.</p>
              </div>
            )}

            {loading && (
              <div className="h-full min-h-[400px] border border-gray-800 rounded-xl flex items-center justify-center text-blue-400 bg-gray-900/30 animate-pulse">
                <div className="flex flex-col items-center gap-4">
                  <Activity className="w-12 h-12" />
                  <p className="font-medium">Extracting claims & querying scholarly sources...</p>
                </div>
              </div>
            )}

            {result && !loading && (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                
                {/* Score Header */}
                <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 flex items-center justify-between shadow-lg">
                  <div>
                    <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-1">Overall Quality</h3>
                    <div className="flex items-baseline gap-2">
                      <span className={`text-5xl font-bold ${result.overall_score >= 70 ? 'text-green-400' : result.overall_score >= 40 ? 'text-yellow-400' : 'text-red-400'}`}>
                        {result.overall_score}
                      </span>
                      <span className="text-xl text-gray-500">/ 100</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-1">Assessment</h3>
                    <span className={`inline-block px-4 py-1.5 rounded-full text-lg font-semibold ${
                      result.overall_assessment === 'Strong' ? 'bg-green-900/30 text-green-400 border border-green-800/50' : 
                      result.overall_assessment === 'Moderate' ? 'bg-yellow-900/30 text-yellow-400 border border-yellow-800/50' : 
                      'bg-red-900/30 text-red-400 border border-red-800/50'
                    }`}>
                      {result.overall_assessment}
                    </span>
                  </div>
                </div>

                {/* Warnings / Fallacies */}
                {(result.detected_fallacies?.length > 0 || result.warnings?.length > 0) && (
                  <div className="bg-red-950/20 border border-red-900/50 rounded-xl p-5 space-y-3">
                    <h3 className="text-red-400 font-semibold flex items-center gap-2">
                      <ShieldAlert className="w-5 h-5" /> Critical Issues Detected
                    </h3>
                    <ul className="space-y-2 text-sm">
                      {result.detected_fallacies.map((f, i) => (
                        <li key={i} className="flex items-start gap-2 text-red-300">
                          <span className="bg-red-900/50 text-red-200 px-2 py-0.5 rounded text-xs font-semibold shrink-0 mt-0.5">{f.type}</span>
                          {f.description}
                        </li>
                      ))}
                      {result.warnings.map((w, i) => (
                        <li key={`w-${i}`} className="flex items-start gap-2 text-yellow-400">
                          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                          {w}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Claims & Structure */}
                  <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-5">
                    <h3 className="font-semibold text-gray-200 border-b border-gray-800 pb-2">Argument Structure</h3>
                    
                    <div>
                      <h4 className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wider">Premises & Claims</h4>
                      <ul className="space-y-2">
                        {result.claims.length === 0 && <p className="text-sm text-gray-600 italic">No explicit claims extracted.</p>}
                        {result.claims.map(c => (
                          <li key={c.id} className="text-sm text-gray-300 bg-gray-950 p-2.5 rounded border border-gray-800/50">
                            {c.text}
                          </li>
                        ))}
                      </ul>
                    </div>

                    {result.conclusions?.length > 0 && (
                      <div>
                        <h4 className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wider">Conclusions</h4>
                        <ul className="space-y-2">
                          {result.conclusions.map((c, i) => (
                            <li key={i} className="text-sm text-indigo-300 bg-indigo-950/20 p-2.5 rounded border border-indigo-900/30 font-medium">
                              {c}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>

                  {/* Component Scores Chart */}
                  <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 flex flex-col">
                    <h3 className="font-semibold text-gray-200 border-b border-gray-800 pb-2 mb-4">Detailed Metrics</h3>
                    <div className="flex-1 min-h-[200px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={Object.entries(result.component_scores).map(([k,v]) => ({name: k.replace(/_/g, ' '), score: v}))} layout="vertical" margin={{top: 0, right: 10, left: -20, bottom: 0}}>
                          <XAxis type="number" domain={[0, 100]} hide />
                          <YAxis dataKey="name" type="category" width={140} stroke="#6B7280" tick={{fontSize: 11, fill: '#9CA3AF'}} axisLine={false} tickLine={false} />
                          <Tooltip 
                            cursor={{fill: '#1F2937'}}
                            contentStyle={{backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '0.5rem', fontSize: '12px'}}
                          />
                          <Bar dataKey="score" fill="#3B82F6" radius={[0, 4, 4, 0]} barSize={12} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                {/* Evidence & Verification */}
                <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
                  <h3 className="font-semibold text-gray-200 border-b border-gray-800 pb-2">Source Verification</h3>
                  
                  {result.verification_results?.length === 0 ? (
                    <p className="text-sm text-gray-500 italic">No evidence detected to verify.</p>
                  ) : (
                    <div className="space-y-4">
                      {result.verification_results.map((vr, idx) => (
                        <VerificationCard key={idx} result={vr} evidence={result.evidence} />
                      ))}
                    </div>
                  )}
                  
                  {/* Claim-Evidence Support */}
                  {result.claim_evidence_relationships?.length > 0 && (
                    <div className="mt-6 pt-4 border-t border-gray-800">
                      <h4 className="text-xs font-semibold text-gray-500 mb-3 uppercase tracking-wider">Internal Reasoning Support</h4>
                      <div className="space-y-2">
                        {result.claim_evidence_relationships.map((rel, i) => (
                          <div key={i} className="text-sm bg-gray-950 border border-gray-800 p-3 rounded-lg">
                            <div className="flex justify-between items-center mb-1">
                              <span className={`font-medium ${
                                rel.support_level === 'Supports' ? 'text-green-400' :
                                rel.support_level === 'Partially supports' ? 'text-yellow-400' : 'text-gray-400'
                              }`}>
                                {rel.support_level}
                              </span>
                            </div>
                            <p className="text-gray-400 text-xs mt-1">{rel.explanation}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

              </div>
            )}
          </section>

        </main>
      </div>
    </div>
  )
}

export default App
