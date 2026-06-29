'use client';

import { motion } from 'framer-motion';
import { ArrowDown, Zap, Layers, Brain, Github } from 'lucide-react';
import DecryptedText from '@/components/ui/DecryptedText';
import ShinyText from '@/components/ui/ShinyText';
import TranslatorSection from '@/components/translator/TranslatorSection';

export default function Home() {
  return (
    <main className="min-h-screen bg-[#050508] overflow-hidden">
      {/* Hero */}
      <section className="relative min-h-screen flex flex-col items-center justify-center px-6">
        {/* Background orbs */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          <div className="absolute top-[20%] left-[15%] w-[400px] h-[400px] bg-indigo-600/10 rounded-full blur-[100px] animate-[pulse-ring_8s_ease-in-out_infinite]" />
          <div className="absolute bottom-[20%] right-[15%] w-[350px] h-[350px] bg-violet-500/8 rounded-full blur-[100px] animate-[pulse-ring_10s_ease-in-out_infinite_1s]" />
          <div className="absolute top-[50%] left-[50%] -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-indigo-900/5 rounded-full blur-[120px]" />
        </div>

        {/* Grid overlay */}
        <div className="absolute inset-0 pointer-events-none opacity-[0.03]"
          style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)', backgroundSize: '60px 60px' }}
        />

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: 'easeOut' }}
          className="relative z-10 text-center max-w-3xl mx-auto"
        >
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-indigo-500/20 bg-indigo-500/5 text-xs text-indigo-300 mb-10 tracking-wide uppercase"
          >
            <Zap size={12} />
            <ShinyText text="Seq2Seq · LSTM · Attention" speed={4} className="text-indigo-300" />
          </motion.div>

          {/* Title */}
          <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold tracking-tight leading-[1.1] mb-6">
            <span className="text-white/90">French → English</span>
            <br />
            <span className="text-indigo-400">
              <DecryptedText
                text="Translator"
                speed={40}
                sequential={true}
                revealDirection="center"
                className="text-indigo-400"
                encryptedClassName="text-indigo-600/50"
                animateOn="view"
              />
            </span>
          </h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
            className="text-base md:text-lg text-white/40 max-w-lg mx-auto mb-12 leading-relaxed"
          >
            Neural machine translation powered by a sequence-to-sequence
            model with attention. Built from scratch in PyTorch.
          </motion.p>

          {/* CTA */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.1 }}
            className="flex items-center justify-center gap-4"
          >
            <a
              href="#translate"
              className="px-7 py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl font-medium text-sm transition-all hover:shadow-[0_0_30px_rgba(99,102,241,0.3)] active:scale-95"
            >
              Try it live
            </a>
            <a
              href="https://github.com/KaranGupta2005/Sequence2Sequence"
              target="_blank"
              className="px-7 py-3 rounded-xl border border-white/10 text-white/60 hover:text-white hover:border-white/20 font-medium text-sm transition-all flex items-center gap-2"
            >
              <Github size={16} />
              Source
            </a>
          </motion.div>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2 }}
          className="absolute bottom-10 left-1/2 -translate-x-1/2 animate-[float_3s_ease-in-out_infinite]"
        >
          <ArrowDown size={20} className="text-white/20" />
        </motion.div>
      </section>

      {/* Translator */}
      <TranslatorSection />

      {/* Architecture Section */}
      <section className="py-24 px-6 border-t border-white/5">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-3">Architecture</h2>
            <p className="text-white/40 text-sm max-w-md mx-auto">How the model processes French input and generates English output</p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              { icon: Brain, title: 'Encoder', desc: 'Bidirectional LSTM processes the full French sentence, capturing context from both directions.', color: 'indigo' },
              { icon: Layers, title: 'Attention', desc: 'Bahdanau attention aligns decoder focus on relevant encoder states at each step.', color: 'violet' },
              { icon: Zap, title: 'Decoder', desc: '2-layer LSTM generates English tokens one at a time using context + previous prediction.', color: 'purple' },
            ].map((item, i) => (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="p-6 rounded-2xl glass glow-border transition-all duration-300"
              >
                <div className={`w-10 h-10 rounded-xl bg-${item.color}-500/10 border border-${item.color}-500/20 flex items-center justify-center mb-4`}>
                  <item.icon size={18} className={`text-${item.color}-400`} />
                </div>
                <h3 className="text-white font-semibold mb-2">{item.title}</h3>
                <p className="text-white/40 text-sm leading-relaxed">{item.desc}</p>
              </motion.div>
            ))}
          </div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-4"
          >
            {[
              { label: 'Sentence Pairs', value: '200K+' },
              { label: 'BLEU Score', value: '0.40' },
              { label: 'Parameters', value: '~25M' },
              { label: 'Layers', value: '2×Bi-LSTM' },
            ].map((stat) => (
              <div key={stat.label} className="text-center p-4 rounded-xl glass">
                <div className="text-xl font-bold text-indigo-400">{stat.value}</div>
                <div className="text-xs text-white/30 mt-1">{stat.label}</div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/5 py-8 px-6">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <span className="text-xs text-white/25">Seq2Seq Translator · PyTorch + Next.js</span>
          <a
            href="https://github.com/KaranGupta2005/Sequence2Sequence"
            target="_blank"
            className="text-white/25 hover:text-white/50 transition-colors"
          >
            <Github size={16} />
          </a>
        </div>
      </footer>
    </main>
  );
}
