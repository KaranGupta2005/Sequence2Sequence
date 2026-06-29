'use client';

import { motion } from 'framer-motion';
import { useMemo } from 'react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

interface BlurTextProps {
    text: string;
    className?: string;
    variant?: {
        hidden: { filter: string; opacity: number; y: number };
        visible: { filter: string; opacity: number; y: number };
    };
    duration?: number;
    delay?: number;
    animateBy?: 'words' | 'letters';
}

export default function BlurText({
    text,
    className = '',
    variant,
    duration = 1,
    delay = 0.2,
    animateBy = 'words',
}: BlurTextProps) {
    const words = useMemo(() => text.split(' '), [text]);
    const letters = useMemo(() => text.split(''), [text]);

    const defaultVariants = {
        hidden: { filter: 'blur(10px)', opacity: 0, y: 20 },
        visible: { filter: 'blur(0px)', opacity: 1, y: 0 },
    };

    const combinedVariants = variant || defaultVariants;

    return (
        <div className={cn("flex flex-wrap", className)}>
            {animateBy === 'words'
                ? words.map((word, i) => (
                    <motion.span
                        key={i}
                        initial="hidden"
                        animate="visible"
                        transition={{ duration, delay: i * delay }}
                        variants={combinedVariants}
                        className="inline-block mr-1"
                    >
                        {word}
                    </motion.span>
                ))
                : letters.map((letter, i) => (
                    <motion.span
                        key={i}
                        initial="hidden"
                        animate="visible"
                        transition={{ duration, delay: i * delay }}
                        variants={combinedVariants}
                        className="inline-block"
                    >
                        {letter === " " ? "\u00A0" : letter}
                    </motion.span>
                ))}
        </div>
    );
}
