import { LayoutDashboard, Zap, Database, BarChart3, Key } from "lucide-react";
import { motion } from "motion/react";
import { useTheme } from "../context/ThemeContext";

interface SidebarProps {
  activePage: string;
  onPageChange: (page: string) => void;
  allowedPages: string[];
}

const menuItems = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "forecast", label: "Forecast", icon: Zap },
  { id: "data", label: "Data", icon: Database },
  { id: "analytics", label: "Analytics", icon: BarChart3 },
];

export function Sidebar({
  activePage,
  onPageChange,
  allowedPages,
}: SidebarProps) {
  const { theme } = useTheme();
  const visibleMenuItems = menuItems.filter((item) =>
    allowedPages.includes(item.id),
  );
  return (
    <motion.aside
      className={`w-[280px] h-screen fixed left-0 top-0 backdrop-blur-xl border-r z-40 transition-colors ${
        theme === "dark"
          ? "bg-[#111827]/95 border-white/10"
          : "bg-white/95 border-gray-200"
      }`}
      initial={{ x: -280 }}
      animate={{ x: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      <div className="flex flex-col h-full py-6">
        {/* Navigation */}
        <nav className="flex-1 px-4 space-y-2 mt-20">
          {visibleMenuItems.map((item, index) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;

            return (
              <motion.button
                key={item.id}
                onClick={() => onPageChange(item.id)}
                className={`
                  w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all relative
                  ${
                    isActive
                      ? "bg-primary/10 text-primary"
                      : theme === "dark"
                        ? "text-foreground-secondary hover:bg-white/5 hover:text-white"
                        : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                  }
                `}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                whileHover={{ x: 4 }}
                whileTap={{ scale: 0.98 }}
              >
                {isActive && (
                  <motion.div
                    className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-primary rounded-r-full shadow-[0_0_12px_rgba(59,130,246,0.8)]"
                    layoutId="activeIndicator"
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                  />
                )}
                <Icon
                  className={`w-5 h-5 ${isActive ? "drop-shadow-[0_0_8px_rgba(59,130,246,0.8)]" : ""}`}
                />
                <span className="font-medium">{item.label}</span>
              </motion.button>
            );
          })}
        </nav>

        {/* Bottom Section */}
        <div className="px-4 mt-auto">
          <div className="backdrop-blur-xl bg-gradient-to-br from-primary/10 to-[#2563EB]/10 rounded-xl p-4 border border-primary/20">
            <div className="flex items-center gap-2 mb-2">
              <Key className="w-4 h-4 text-primary" />
              <span className="text-xs font-semibold text-primary">
                API Access
              </span>
            </div>
            <p
              className={`text-xs mb-3 ${
                theme === "dark" ? "text-foreground-secondary" : "text-gray-600"
              }`}
            >
              Connect your systems with our REST API
            </p>
            <button className="w-full py-2 px-3 rounded-lg bg-primary/20 hover:bg-primary/30 text-primary text-xs font-medium transition-colors border border-primary/30">
              View Documentation
            </button>
          </div>
        </div>
      </div>
    </motion.aside>
  );
}
