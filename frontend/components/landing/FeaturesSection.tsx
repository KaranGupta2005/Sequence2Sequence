'use client';

import React from "react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { Brain, Zap, Layers, ArrowRight } from "lucide-react";

export default function FeaturesSection() {
    const features = [
        {
            title: "Sequence-to-Sequence Architecture",
            description:
                "An Encoder-Decoder LSTM architecture that learns to map variable-length French input sequences to English output sequences, capturing complex language structure.",
            skeleton: <SkeletonOne />,
            className: "col-span-1 lg:col-span-4 border-b lg:border-r border-white/10",
        },
        {
            title: "Bahdanau Attention",
            description:
                "The attention mechanism allows the decoder to focus on different parts of the input sentence at each decoding step, dramatically improving translation quality.",
            skeleton: <SkeletonTwo />,
            className: "border-b col-span-1 lg:col-span-2 border-white/10",
        },
        {
            title: "Teacher Forcing Training",
            description:
                "During training, the model uses ground-truth tokens as decoder inputs. This accelerates convergence and produces more fluent translations.",
            skeleton: <SkeletonThree />,
            className: "col-span-1 lg:col-span-3 lg:border-r border-white/10",
        },
        {
            title: "Trained on 200K+ Pairs",
            description:
                "The model is trained on the Tatoeba French-English parallel corpus with over 200,000 sentence pairs, covering everyday conversations and common phrases.",
            skeleton: <SkeletonFour />,
            className: "col-span-1 lg:col-span-3 border-b lg:border-none border-white/10",
        },
    ];

    return (
        <div id="features" className="relative z-20 py-20 lg:py-40 max-w-7xl mx-auto">
            <div className="px-8 text-center">
                <h4 className="text-3xl lg:text-5xl lg:leading-tight max-w-5xl mx-auto tracking-tight font-bold text-white mb-6">
                    How It <span className="text-cyan-400">Works</span>
                </h4>

                <p className="text-lg max-w-2xl mx-auto text-white/50">
                    A deep dive into the neural architecture that powers this translator.
                    Built entirely from scratch using PyTorch.
                </p>
            </div>

            <div className="relative">
                <div className="grid grid-cols-1 lg:grid-cols-6 mt-16 xl:border rounded-3xl border-white/10 bg-black/20 overflow-hidden">
                    {features.map((feature) => (
                        <FeatureCard key={feature.title} className={feature.className}>
                            <FeatureTitle>{feature.title}</FeatureTitle>
                            <FeatureDescription>{feature.description}</FeatureDescription>
                            <div className="h-full w-full">{feature.skeleton}</div>
                        </FeatureCard>
                    ))}
                </div>
            </div>
        </div>
    );
}

const FeatureCard = ({
    children,
    className,
}: {
    children?: React.ReactNode;
    className?: string;
}) => {
    return (
        <div className={cn(`p-8 relative overflow-hidden`, className)}>
            {children}
        </div>
    );
};

const FeatureTitle = ({ children }: { children?: React.ReactNode }) => {
    return (
        <p className="max-w-5xl mx-auto text-left tracking-tight text-white text-2xl font-bold mb-3">
            {children}
        </p>
    );
};

const FeatureDescription = ({ children }: { children?: React.ReactNode }) => {
    return (
        <p className={cn(
            "text-base text-left max-w-4xl text-white/50",
            "text-left max-w-lg mx-0 mb-8"
        )}>
            {children}
        </p>
    );
};

// Skeleton 1: Encoder-Decoder Architecture Visualization
const SkeletonOne = () => {
    return (
        <div className="relative flex py-4 px-2 gap-10 h-full">
            <div className="w-full p-6 mx-auto bg-black/40 border border-white/10 rounded-xl shadow-2xl group h-full relative overflow-hidden">
                <div className="absolute top-4 right-4 flex gap-2">
                    <div className="w-2 h-2 rounded-full bg-red-500/50" />
                    <div className="w-2 h-2 rounded-full bg-yellow-500/50" />
                    <div className="w-2 h-2 rounded-full bg-green-500/50" />
                </div>

                <div className="mt-4 space-y-4">
                    <div className="space-y-2">
                        <div className="flex justify-between text-xs text-white/60">
                            <span>Encoder LSTM</span>
                            <span>512 hidden</span>
                        </div>
                        <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                whileInView={{ width: "100%" }}
                                transition={{ duration: 1.5, delay: 0.2 }}
                                className="h-full bg-cyan-500"
                            />
                        </div>
                    </div>

                    <div className="space-y-2">
                        <div className="flex justify-between text-xs text-white/60">
                            <span>Attention Layer</span>
                            <span>512 → 1</span>
                        </div>
                        <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                whileInView={{ width: "75%" }}
                                transition={{ duration: 1.5, delay: 0.4 }}
                                className="h-full bg-blue-500"
                            />
                        </div>
                    </div>

                    <div className="space-y-2">
                        <div className="flex justify-between text-xs text-white/60">
                            <span>Decoder LSTM</span>
                            <span>768 → 512</span>
                        </div>
                        <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                whileInView={{ width: "88%" }}
                                transition={{ duration: 1.5, delay: 0.6 }}
                                className="h-full bg-purple-500"
                            />
                        </div>
                    </div>

                    <div className="space-y-2">
                        <div className="flex justify-between text-xs text-white/60">
                            <span>Embedding</span>
                            <span>256 dim</span>
                        </div>
                        <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                whileInView={{ width: "60%" }}
                                transition={{ duration: 1.5, delay: 0.8 }}
                                className="h-full bg-green-500"
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

// Skeleton 2: Attention Weights Visualization
const SkeletonTwo = () => {
    return (
        <div className="relative flex flex-col items-center justify-center p-8 h-full w-full">
            <div className="grid grid-cols-4 gap-1 relative z-10">
                {[0.9, 0.2, 0.1, 0.3, 0.1, 0.8, 0.3, 0.1, 0.2, 0.1, 0.7, 0.2, 0.1, 0.3, 0.2, 0.8].map((weight, i) => (
                    <motion.div
                        key={i}
                        initial={{ opacity: 0, scale: 0 }}
                        whileInView={{ opacity: weight, scale: 1 }}
                        transition={{ delay: i * 0.05, duration: 0.3 }}
                        className="w-8 h-8 rounded-md bg-cyan-400"
                    />
                ))}
            </div>
            <div className="mt-4 flex gap-6 text-[10px] text-white/40">
                <span>Source →</span>
                <span>↓ Target</span>
            </div>
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 bg-cyan-500/10 rounded-full blur-3xl" />
        </div>
    );
};

// Skeleton 3: Teacher Forcing
const SkeletonThree = () => {
    return (
        <div className="relative flex gap-4 h-full group p-4">
            <div className="flex flex-col gap-3 w-full">
                <div className="bg-white/5 border border-white/10 rounded-2xl rounded-tl-sm p-4 text-sm text-white/80">
                    <span className="text-cyan-400 font-bold block mb-1">Training Step</span>
                    Input: <code className="bg-white/10 px-1 rounded text-xs">je suis</code>
                    <br />
                    Target: <code className="bg-white/10 px-1 rounded text-xs">i am</code>
                </div>

                <div className="bg-cyan-500/10 border border-cyan-500/20 rounded-2xl rounded-tl-sm p-4 text-sm text-white/90 self-start">
                    <span className="text-green-400 font-bold block mb-1">Teacher Forcing</span>
                    Feeds ground-truth &quot;i&quot; → predicts &quot;am&quot; ✓
                </div>
            </div>
        </div>
    );
};

// Skeleton 4: Training Data Stats
const SkeletonFour = () => {
    const stats = [
        { label: "Sentence Pairs", value: "200K+" },
        { label: "French Vocab", value: "~12K" },
        { label: "English Vocab", value: "~8K" },
        { label: "Epochs", value: "10" },
    ];

    return (
        <div className="grid grid-cols-2 gap-4 p-4">
            {stats.map((stat, i) => (
                <motion.div
                    key={stat.label}
                    initial={{ opacity: 0, y: 10 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.1 }}
                    className="bg-white/5 border border-white/10 rounded-xl p-4 text-center"
                >
                    <div className="text-2xl font-bold text-cyan-400">{stat.value}</div>
                    <div className="text-xs text-white/40 mt-1">{stat.label}</div>
                </motion.div>
            ))}
        </div>
    );
};
