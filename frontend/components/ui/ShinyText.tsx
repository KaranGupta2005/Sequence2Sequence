'use client';

interface ShinyTextProps {
  text: string;
  disabled?: boolean;
  speed?: number;
  className?: string;
}

export default function ShinyText({ text, disabled = false, speed = 5, className = '' }: ShinyTextProps) {
  return (
    <span
      className={`inline-block bg-clip-text text-transparent bg-[length:250%_100%] ${disabled ? '' : 'animate-[shine_linear_infinite]'} ${className}`}
      style={{
        backgroundImage: 'linear-gradient(120deg, rgba(255,255,255,0.4) 40%, rgba(255,255,255,1) 50%, rgba(255,255,255,0.4) 60%)',
        animationDuration: `${speed}s`,
      }}
    >
      {text}
    </span>
  );
}
