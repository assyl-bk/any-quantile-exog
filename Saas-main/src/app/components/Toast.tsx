import { motion, AnimatePresence } from "motion/react";
import { CheckCircle, AlertCircle, Info, X } from "lucide-react";
import { useEffect, useState } from "react";

interface ToastProps {
  message: string;
  type?: "success" | "error" | "info";
  duration?: number;
  onClose?: () => void;
}

export function Toast({ message, type = "info", duration = 3000, onClose }: ToastProps) {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false);
      setTimeout(() => onClose?.(), 300);
    }, duration);

    return () => clearTimeout(timer);
  }, [duration, onClose]);

  const iconMap = {
    success: <CheckCircle className="w-5 h-5 text-success" />,
    error: <AlertCircle className="w-5 h-5 text-critical" />,
    info: <Info className="w-5 h-5 text-primary" />,
  };

  const colorMap = {
    success: "border-success/30 bg-success/10",
    error: "border-critical/30 bg-critical/10",
    info: "border-primary/30 bg-primary/10",
  };

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          className={`fixed top-24 right-8 z-50 backdrop-blur-xl border rounded-xl px-6 py-4 shadow-2xl ${colorMap[type]}`}
          initial={{ opacity: 0, y: -20, x: 100 }}
          animate={{ opacity: 1, y: 0, x: 0 }}
          exit={{ opacity: 0, x: 100 }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
        >
          <div className="flex items-center gap-3 min-w-[300px]">
            {iconMap[type]}
            <p className="text-white font-medium flex-1">{message}</p>
            <button
              onClick={() => {
                setIsVisible(false);
                setTimeout(() => onClose?.(), 300);
              }}
              className="p-1 rounded-lg hover:bg-white/10 transition-colors"
            >
              <X className="w-4 h-4 text-foreground-secondary" />
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// Toast Container for multiple toasts
export function ToastContainer({ toasts }: { toasts: Array<{ id: string } & ToastProps> }) {
  return (
    <div className="fixed top-24 right-8 z-50 space-y-2">
      {toasts.map((toast, index) => (
        <motion.div
          key={toast.id}
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          <Toast {...toast} />
        </motion.div>
      ))}
    </div>
  );
}
