'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { Languages, ArrowRight, Sparkles, Brain, Cpu, Zap, Volume2, Copy, Check, RotateCcw } from 'lucide-react';
import GradientText from '@/components/ui/GradientText';
import BlurText from '@/components/ui/BlurText';
import SpotlightCard from '@/components/ui/SpotlightCard';
import TranslatorSection from '@/components/translator/TranslatorSection';
import FeaturesSection from '@/components/landing/FeaturesSection';
import Footer from '@/components/landing/Footer';

export default function Home() {
  return (
    <main className="min-h-screen bg-[#09090b]">
      {/* Hero Section */}
      <section className="relative min-h-screen flex flex-col items-center justify-center text-center px-6 py-20 overflow-hidden">
        {/* Dynamic Background */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-cyan-600/20 rounded-full blur-[120px] animate-[aurora_15s_infinite]" />
          <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-blue-500/10 rounded-full blur-[120px] animate-[aurora_15s_infinite_reverse]" />
        </div>

        {/* Floating Decorations */}
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 1, duration: 1.5 }}
          className="absolute left-[10%] top-[30%] hidden lg:block p-4 glass rounded-xl border-l-4 border-cyan-500"
        >
          <Languages className="text-cyan-400 mb-2" size={24} />
          <div className="space-y-2">
            <div className="text-xs text-white/60 font-mono">Bonjour le monde</div>
            <div className="w-24 h-2 bg-white/10 rounded" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 1.2, duration: 1.5 }}
          className="absolute right-[10%] bottom-[40%] hidden lg:block p-4 glass rounded-xl border-l-4 border-blue-500"
        >
          <Brain className="text-blue-400 mb-2" size={24} />
          <div className="space-y-2">
            <div className="text-xs text-white/60 font-mono">Hello world</div>
            <div className="w-28 h-2 bg-white/10 rounded" />
          </div>
        </motion.div>

        {/* Main Content */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="relative z-10 max-w-4xl mx-auto"
        >
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5, duration: 0.5 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border-cyan-500/30 text-sm text-cyan-300 mb-8"
          >
            <Sparkles size={14} className="animate-pulse" />
            <span>LSTM + Attention · Seq2Seq Architecture</span>
          </motion.div>

          {/* Headline */}
          <h1 className="text-6xl md:text-8xl font-bold tracking-tight mb-8 leading-tight">
            <span className="text-white">French to English</span>
            <br />
            <div className="flex justify-center">
              <GradientText
                colors={["#06b6d4", "#3b82f6", "#06b6d4", "#3b82f6", "#06b6d4"]}
                animationSpeed={6}
                showBorder={false}
                className="text-6xl md:text-8xl font-bold tracking-tight"
              >
                Translator
              </GradientText>
            </div>
          </h1>

          <p className="text-lg md:text-xl text-white/50 max-w-2xl mx-auto mb-12">
            Neural Machine Translation powered by a Sequence-to-Sequence model
            <br className="hidden md:block" />
            with Bahdanau Attention mechanism. Built from scratch with PyTorch.
          </p>

          {/* CTA */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
            <a
              href="#translate"
              className="group relative px-8 py-4 bg-white text-black rounded-full font-bold text-lg overflow-hidden transition-transform hover:scale-105"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-blue-400 opacity-0 group-hover:opacity-100 transition-opacity" />
              <span className="relative flex items-center gap-2">
                Start Translating
                <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
              </span>
            </a>

            <a
              href="#features"
              className="px-8 py-4 rounded-full glass hover:bg-white/5 text-white font-medium text-lg transition-all hover:scale-105"
            >
              How It Works
            </a>
          </div>
        </motion.div>
      </section>

      {/* Translator Section */}
      <TranslatorSection />

      {/* Features Section */}
      <FeaturesSection />

      {/* Footer */}
      <Footer />
    </main>
  );
}
