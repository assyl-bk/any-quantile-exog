import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Search, Zap, Database, BarChart3, Settings, Calendar, Upload } from "lucide-react";

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigate: (page: string) => void;
  allowedPages: string[];
}

const commands = [
  { id: "dashboard", label: "Go to Dashboard", icon: BarChart3, category: "Navigation" },
  { id: "forecast", label: "Generate Forecast", icon: Zap, category: "Navigation" },
  { id: "data", label: "Manage Data", icon: Database, category: "Navigation" },
  { id: "analytics", label: "View Analytics", icon: BarChart3, category: "Navigation" },
  { id: "settings", label: "Open Settings", icon: Settings, category: "Navigation" },
  { id: "upload", label: "Upload Dataset", icon: Upload, category: "Actions" },
  { id: "new-forecast", label: "New Forecast", icon: Calendar, category: "Actions" },
];

export function CommandPalette({ isOpen, onClose, onNavigate, allowedPages }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);

  const filteredCommands = commands.filter((cmd) => {
    const isNav = cmd.category === "Navigation";
    if (isNav && !allowedPages.includes(cmd.id)) {
      return false;
    }
    return cmd.label.toLowerCase().includes(query.toLowerCase());
  });

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((prev) => (prev + 1) % filteredCommands.length);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((prev) => (prev - 1 + filteredCommands.length) % filteredCommands.length);
      } else if (e.key === "Enter" && filteredCommands[selectedIndex]) {
        e.preventDefault();
        handleSelect(filteredCommands[selectedIndex].id);
      }
    };

    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
    }

    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, filteredCommands, selectedIndex]);

  const handleSelect = (commandId: string) => {
    const navCommands = ["dashboard", "forecast", "data", "analytics", "settings"];
    if (navCommands.includes(commandId)) {
      if (!allowedPages.includes(commandId)) {
        return;
      }
      onNavigate(commandId);
    }
    onClose();
    setQuery("");
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <div className="fixed inset-0 flex items-start justify-center pt-[20vh] z-50 pointer-events-none">
            <motion.div
              className="w-full max-w-2xl pointer-events-auto"
              initial={{ opacity: 0, scale: 0.95, y: -20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: -20 }}
              transition={{ type: "spring", stiffness: 300, damping: 30 }}
            >
              <div className="backdrop-blur-xl bg-[#111827]/95 border border-white/20 rounded-2xl shadow-2xl overflow-hidden">
                {/* Search Input */}
                <div className="flex items-center gap-3 p-4 border-b border-white/10">
                  <Search className="w-5 h-5 text-foreground-secondary" />
                  <input
                    type="text"
                    placeholder="Type a command or search..."
                    value={query}
                    onChange={(e) => {
                      setQuery(e.target.value);
                      setSelectedIndex(0);
                    }}
                    className="flex-1 bg-transparent text-white placeholder:text-muted-foreground focus:outline-none"
                    autoFocus
                  />
                  <kbd className="px-2 py-1 rounded bg-white/10 text-xs text-foreground-secondary border border-white/20">
                    ESC
                  </kbd>
                </div>

                {/* Commands List */}
                <div className="max-h-[400px] overflow-y-auto">
                  {filteredCommands.length === 0 ? (
                    <div className="p-8 text-center">
                      <p className="text-foreground-secondary">No results found</p>
                    </div>
                  ) : (
                    <div className="p-2">
                      {["Navigation", "Actions"].map((category) => {
                        const categoryCommands = filteredCommands.filter(
                          (cmd) => cmd.category === category
                        );
                        if (categoryCommands.length === 0) return null;

                        return (
                          <div key={category} className="mb-2">
                            <p className="px-3 py-2 text-xs font-semibold text-foreground-secondary uppercase tracking-wider">
                              {category}
                            </p>
                            <div className="space-y-1">
                              {categoryCommands.map((cmd, index) => {
                                const Icon = cmd.icon;
                                const globalIndex = filteredCommands.indexOf(cmd);
                                const isSelected = globalIndex === selectedIndex;

                                return (
                                  <button
                                    key={cmd.id}
                                    onClick={() => handleSelect(cmd.id)}
                                    className={`w-full flex items-center gap-3 px-3 py-3 rounded-lg transition-all ${
                                      isSelected
                                        ? "bg-primary/20 text-primary border border-primary/30"
                                        : "text-foreground-secondary hover:bg-white/5"
                                    }`}
                                  >
                                    <Icon className="w-5 h-5" />
                                    <span className="font-medium">{cmd.label}</span>
                                    {isSelected && (
                                      <kbd className="ml-auto px-2 py-1 rounded bg-primary/20 text-xs border border-primary/30">
                                        ↵
                                      </kbd>
                                    )}
                                  </button>
                                );
                              })}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>

                {/* Footer */}
                <div className="px-4 py-3 border-t border-white/10 flex items-center justify-between text-xs text-foreground-secondary">
                  <div className="flex gap-4">
                    <span className="flex items-center gap-1">
                      <kbd className="px-1.5 py-0.5 rounded bg-white/10 border border-white/20">↑</kbd>
                      <kbd className="px-1.5 py-0.5 rounded bg-white/10 border border-white/20">↓</kbd>
                      Navigate
                    </span>
                    <span className="flex items-center gap-1">
                      <kbd className="px-1.5 py-0.5 rounded bg-white/10 border border-white/20">↵</kbd>
                      Select
                    </span>
                  </div>
                  <span>Press ESC to close</span>
                </div>
              </div>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
}
