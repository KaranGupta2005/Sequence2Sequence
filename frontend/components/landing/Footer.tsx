'use client';

import { Languages, Github, Brain } from 'lucide-react';
import GradientText from '@/components/ui/GradientText';

export default function Footer() {
    return (
        <footer className="relative border-t border-white/10 py-12 px-6">
            <div className="max-w-7xl mx-auto">
                <div className="flex flex-col md:flex-row items-center justify-between gap-8">
                    {/* Logo/Brand */}
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                            <Languages size={20} className="text-white" />
                        </div>
                        <div>
                            <h3 className="text-white font-bold text-lg">Seq2Seq Translator</h3>
                            <p className="text-white/40 text-xs">Neural Machine Translation</p>
                        </div>
                    </div>

                    {/* Tech Stack */}
                    <div className="flex items-center gap-6 text-sm text-white/40">
                        <span className="flex items-center gap-1.5">
                            <Brain size={14} className="text-cyan-400" />
                            PyTorch
                        </span>
                        <span>•</span>
                        <span>FastAPI</span>
                        <span>•</span>
                        <span>Next.js</span>
                        <span>•</span>
                        <span>LSTM + Attention</span>
                    </div>

                    {/* Links */}
                    <div className="flex items-center gap-4">
                        <a
                            href="https://github.com/KaranGupta2005/Sequence2Sequence"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="p-2 rounded-lg hover:bg-white/5 text-white/40 hover:text-white transition-colors"
                        >
                            <Github size={20} />
                        </a>
                    </div>
                </div>

                <div className="mt-8 pt-8 border-t border-white/5 text-center text-xs text-white/30">
                    Built with ♥ using Sequence-to-Sequence with Attention • French → English Translation
                </div>
            </div>
        </footer>
    );
}
