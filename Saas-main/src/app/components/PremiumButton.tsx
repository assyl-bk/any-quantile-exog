import { motion } from "motion/react";
import { ReactNode } from "react";

interface PremiumButtonProps {
  children: ReactNode;
  variant?: "primary" | "secondary" | "danger" | "ghost";
  size?: "sm" | "md" | "lg";
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
  icon?: ReactNode;
}

export function PremiumButton({
  children,
  variant = "primary",
  size = "md",
  onClick,
  disabled = false,
  className = "",
  icon,
}: PremiumButtonProps) {
  const baseClasses = "rounded-xl font-semibold transition-all flex items-center justify-center gap-2";
  
  const variantClasses = {
    primary: "bg-gradient-to-r from-primary to-[#2563EB] text-white hover:shadow-lg hover:shadow-primary/30",
    secondary: "bg-white/5 text-foreground-secondary hover:bg-white/10 border border-white/10",
    danger: "bg-gradient-to-r from-critical to-red-600 text-white hover:shadow-lg hover:shadow-critical/30",
    ghost: "bg-transparent text-foreground-secondary hover:bg-white/5",
  };

  const sizeClasses = {
    sm: "px-4 py-2 text-sm",
    md: "px-6 py-3 text-base",
    lg: "px-8 py-4 text-lg",
  };

  return (
    <motion.button
      className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${
        disabled ? "opacity-50 cursor-not-allowed" : ""
      } ${className}`}
      onClick={onClick}
      disabled={disabled}
      whileHover={!disabled ? { scale: 1.02 } : {}}
      whileTap={!disabled ? { scale: 0.98 } : {}}
    >
      {icon && <span>{icon}</span>}
      {children}
    </motion.button>
  );
}
