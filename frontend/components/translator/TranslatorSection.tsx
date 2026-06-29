'use client';

import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Copy, Check, RotateCcw, Loader2, ArrowRight, Sparkles, AlertCircle } from 'lucide-react';
import SpotlightCard from '@/components/ui/SpotlightCard';

interface TranslationResult {
  input_text: string;
  translated_text: string;
  cleaned_input: string;
}

interface Example {
  french: string;
  expected: string;
}

export default function TranslatorSection() {
  const [input, setInput] = useState('');
  const [result, setResult] = useState<TranslationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [history, setHistory] = useState<TranslationResult[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const examples: Example[] = [
    { french: "Bonjour, comment allez-vous?", expected: "Hello, how are you?" },
    { french: "Je suis un étudiant.", expected: "I am a student." },
    { french: "Il fait beau aujourd'hui.", expected: "It is nice today." },
    { french: "Merci beaucoup.", expected: "Thank you very much." },
    { french: "Je t'aime.", expected: "I love you." },
    { french: "Où est la gare?", expected: "Where is the station?" },
    { french: "J'ai faim.", expected: "I am hungry." },
    { french: "Bonne nuit.", expected: "Good night." },
  ];

  const handleTranslate = async (textOverride?: string) => {
    const textToTranslate = textOverride || input;
    if (!textToTranslate.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || '';
      const response = await fetch(`${apiBase}/api/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textToTranslate, max_length: 50 }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Translation failed (${response.status})`);
      }

      const data: TranslationResult = await response.json();
      setResult(data);
      setHistory(prev => [data, ...prev.slice(0, 9)]);
    } catch (err: any) {
      setError(err.message || 'Translation failed. Make sure the backend is running.');
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
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  const handleExampleClick = (example: Example) => {
    setInput(example.french);
    handleTranslate(example.french);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleTranslate();
    }
  };

  return (
    <section id="translate" className="relative py-20 px-6">
      <div className="max-w-5xl mx-auto">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Try It <span className="text-cyan-400">Live</span>
          </h2>
          <p className="text-white/50 text-lg max-w-xl mx-auto">
            Type French text below and watch the neural network translate it in real-time.
          </p>
        </motion.div>

        {/* Translator Card */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 }}
        >
          <SpotlightCard className="!p-0 overflow-hidden" spotlightColor="rgba(6, 182, 212, 0.15)">
            <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-white/10">
              {/* Input Side */}
              <div className="p-6 md:p-8">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-blue-500" />
                    <span className="text-sm font-medium text-white/70">French</span>
                  </div>
                  <button
                    onClick={handleClear}
                    className="p-2 rounded-lg hover:bg-white/5 text-white/40 hover:text-white/70 transition-colors"
                    title="Clear"
                  >
                    <RotateCcw size={16} />
                  </button>
                </div>

                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Tapez votre texte en français..."
                  className="w-full bg-transparent text-white text-lg resize-none focus:outline-none min-h-[150px] placeholder:text-white/20"
                  rows={5}
                />

                <div className="flex items-center justify-between mt-4 pt-4 border-t border-white/5">
                  <span className="text-xs text-white/30">
                    {input.length > 0 ? `${input.split(' ').filter(Boolean).length} words` : 'Press Enter to translate'}
                  </span>
                  <button
                    onClick={() => handleTranslate()}
                    disabled={!input.trim() || isLoading}
                    className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-cyan-600 to-blue-600 text-white rounded-xl font-medium text-sm disabled:opacity-40 disabled:cursor-not-allowed hover:brightness-110 transition-all active:scale-95"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 size={16} className="animate-spin" />
                        Translating...
                      </>
                    ) : (
                      <>
                        <Send size={16} />
                        Translate
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Output Side */}
              <div className="p-6 md:p-8 bg-white/[0.02]">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-cyan-500" />
                    <span className="text-sm font-medium text-white/70">English</span>
                  </div>
                  {result && (
                    <button
                      onClick={handleCopy}
                      className="p-2 rounded-lg hover:bg-white/5 text-white/40 hover:text-white/70 transition-colors"
                      title="Copy translation"
                    >
                      {copied ? <Check size={16} className="text-green-400" /> : <Copy size={16} />}
                    </button>
                  )}
                </div>

                <div className="min-h-[150px] flex items-start">
                  <AnimatePresence mode="wait">
                    {error ? (
                      <motion.div
                        key="error"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className="flex items-start gap-3 text-red-400"
                      >
                        <AlertCircle size={20} className="mt-0.5 shrink-0" />
                        <p className="text-sm">{error}</p>
                      </motion.div>
                    ) : result ? (
                      <motion.p
                        key="result"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className="text-lg text-white leading-relaxed"
                      >
                        {result.translated_text}
                      </motion.p>
                    ) : isLoading ? (
                      <motion.div
                        key="loading"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="flex items-center gap-2 text-white/30"
                      >
                        <div className="flex space-x-1">
                          <div className="w-1.5 h-1.5 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                          <div className="w-1.5 h-1.5 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                          <div className="w-1.5 h-1.5 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                        </div>
                        <span className="text-sm">Processing...</span>
                      </motion.div>
                    ) : (
                      <motion.p
                        key="placeholder"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="text-white/20 text-lg"
                      >
                        Translation will appear here...
                      </motion.p>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            </div>
          </SpotlightCard>
        </motion.div>

        {/* Example Chips */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.4 }}
          className="mt-8"
        >
          <p className="text-center text-sm text-white/30 mb-4">Try these examples:</p>
          <div className="flex flex-wrap justify-center gap-2">
            {examples.map((example, i) => (
              <button
                key={i}
                onClick={() => handleExampleClick(example)}
                className="px-4 py-2 rounded-full glass text-sm text-white/60 hover:text-white hover:bg-white/5 transition-all hover:scale-105 active:scale-95"
              >
                {example.french}
              </button>
            ))}
          </div>
        </motion.div>

        {/* Translation History */}
        <AnimatePresence>
          {history.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="mt-12"
            >
              <h3 className="text-lg font-semibold text-white/70 mb-4 flex items-center gap-2">
                <RotateCcw size={16} />
                Recent Translations
              </h3>
              <div className="space-y-3">
                {history.slice(0, 5).map((item, i) => (
                  <motion.div
                    key={`${item.input_text}-${i}`}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05 }}
                    className="flex items-center gap-4 p-4 rounded-xl glass"
                  >
                    <span className="text-sm text-white/50 flex-1 truncate">{item.input_text}</span>
                    <ArrowRight size={14} className="text-cyan-500 shrink-0" />
                    <span className="text-sm text-white flex-1 truncate">{item.translated_text}</span>
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
