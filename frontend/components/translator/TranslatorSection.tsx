'use client';

import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Copy, Check, RotateCcw, Loader2, ArrowRight, AlertCircle } from 'lucide-react';

interface TranslationResult {
  input_text: string;
  translated_text: string;
  cleaned_input: string;
}

export default function TranslatorSection() {
  const [input, setInput] = useState('');
  const [result, setResult] = useState<TranslationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [history, setHistory] = useState<TranslationResult[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const examples = [
    "Bonjour, comment allez-vous?",
    "Je suis un étudiant.",
    "Merci beaucoup.",
    "Je t'aime.",
    "Il fait froid.",
    "Bonne nuit.",
    "Où est la gare?",
    "Je ne comprends pas.",
  ];

  const handleTranslate = async (textOverride?: string) => {
    const textToTranslate = textOverride || input;
    if (!textToTranslate.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textToTranslate, max_length: 30 }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Failed (${response.status})`);
      }

      const data: TranslationResult = await response.json();
      setResult(data);
      setHistory(prev => [data, ...prev.slice(0, 4)]);
    } catch (err: any) {
      setError(err.message || 'Translation failed. Is the backend running?');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = () => {
    if (result?.translated_text) {
      navigator.clipboard.writeText(result.translated_text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleClear = () => {
    setInput('');
    setResult(null);
    setError(null);
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleTranslate();
    }
  };

  return (
    <section id="translate" className="py-24 px-6">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-10"
        >
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-3">Try it live</h2>
          <p className="text-white/40 text-sm">Type French text and get instant English translation</p>
        </motion.div>

        {/* Translation Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.1 }}
          className="rounded-2xl glass glow-border overflow-hidden"
        >
          {/* Input */}
          <div className="p-6 border-b border-white/5">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-medium text-white/50 uppercase tracking-wider">French</span>
              <button onClick={handleClear} className="p-1.5 rounded-lg hover:bg-white/5 text-white/30 hover:text-white/60 transition">
                <RotateCcw size={14} />
              </button>
            </div>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Tapez en français..."
              className="w-full bg-transparent text-white text-base resize-none focus:outline-none min-h-[80px] placeholder:text-white/15 leading-relaxed"
              rows={3}
            />
            <div className="flex items-center justify-between mt-3 pt-3 border-t border-white/5">
              <span className="text-[11px] text-white/20">
                {input ? `${input.split(' ').filter(Boolean).length} words` : 'Enter to translate'}
              </span>
              <button
                onClick={() => handleTranslate()}
                disabled={!input.trim() || isLoading}
                className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-white/5 disabled:text-white/20 text-white rounded-lg text-xs font-medium transition-all active:scale-95 disabled:cursor-not-allowed"
              >
                {isLoading ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
                {isLoading ? 'Translating...' : 'Translate'}
              </button>
            </div>
          </div>

          {/* Output */}
          <div className="p-6 bg-white/[0.01]">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-medium text-white/50 uppercase tracking-wider">English</span>
              {result && (
                <button onClick={handleCopy} className="p-1.5 rounded-lg hover:bg-white/5 text-white/30 hover:text-white/60 transition">
                  {copied ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
                </button>
              )}
            </div>
            <div className="min-h-[80px] flex items-start">
              <AnimatePresence mode="wait">
                {error ? (
                  <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    className="flex items-start gap-2 text-red-400/80 text-sm">
                    <AlertCircle size={16} className="mt-0.5 shrink-0" />
                    <p>{error}</p>
                  </motion.div>
                ) : result ? (
                  <motion.p key="result" initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    className="text-base text-white leading-relaxed">
                    {result.translated_text}
                  </motion.p>
                ) : isLoading ? (
                  <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    className="flex items-center gap-2 text-white/20 text-sm">
                    <div className="flex gap-1">
                      <div className="w-1 h-1 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-1 h-1 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-1 h-1 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </motion.div>
                ) : (
                  <motion.p key="placeholder" className="text-white/15 text-base">
                    Translation appears here...
                  </motion.p>
                )}
              </AnimatePresence>
            </div>
          </div>
        </motion.div>

        {/* Examples */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3 }}
          className="mt-6"
        >
          <p className="text-center text-[11px] text-white/20 mb-3 uppercase tracking-wider">Examples</p>
          <div className="flex flex-wrap justify-center gap-1.5">
            {examples.map((ex, i) => (
              <button
                key={i}
                onClick={() => { setInput(ex); handleTranslate(ex); }}
                className="px-3 py-1.5 rounded-lg border border-white/5 text-[11px] text-white/40 hover:text-white/70 hover:border-white/10 hover:bg-white/[0.02] transition-all"
              >
                {ex}
              </button>
            ))}
          </div>
        </motion.div>

        {/* History */}
        <AnimatePresence>
          {history.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="mt-10"
            >
              <p className="text-[11px] text-white/20 uppercase tracking-wider mb-3">Recent</p>
              <div className="space-y-2">
                {history.map((item, i) => (
                  <motion.div
                    key={`${item.input_text}-${i}`}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.03 }}
                    className="flex items-center gap-3 p-3 rounded-xl border border-white/5 bg-white/[0.01]"
                  >
                    <span className="text-xs text-white/40 flex-1 truncate">{item.input_text}</span>
                    <ArrowRight size={12} className="text-indigo-500/50 shrink-0" />
                    <span className="text-xs text-white/70 flex-1 truncate">{item.translated_text}</span>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
