import { useState, useEffect } from "react";
import { Header } from "./components/Header";
import { Sidebar } from "./components/Sidebar";
import { Dashboard } from "./components/Dashboard";
import { ForecastPage } from "./components/ForecastPage";
import { DataPage } from "./components/DataPage";
import { AnalyticsPage } from "./components/AnalyticsPage";
import { SettingsPage } from "./components/SettingsPage";
import { CommandPalette } from "./components/CommandPalette";
import { AuthPage } from "./components/AuthPage";
import { ThemeProvider, useTheme } from "./context/ThemeContext";
import { AuthProvider, useAuth, UserRole } from "./context/AuthContext";
import { DataProvider } from "./context/Datacontext";
import { motion, AnimatePresence } from "motion/react";

const PAGE_ACCESS: Record<UserRole, string[]> = {
  energy_grid_operator: [
    "dashboard",
    "forecast",
    "data",
    "analytics",
    "settings",
  ],
  energy_trader: ["dashboard", "forecast", "data", "analytics", "settings"],
  energy_planner: ["dashboard", "forecast", "data", "analytics", "settings"],
  system_administrator: [
    "dashboard",
    "forecast",
    "data",
    "analytics",
    "settings",
  ],
};

function AppContent() {
  const [currentPage, setCurrentPage] = useState("dashboard");
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const { theme } = useTheme();
  const { isAuthenticated, isLoading, user } = useAuth();
  const defaultRole: UserRole = "energy_grid_operator";
  const currentRole: UserRole = user?.role ?? defaultRole;
  const allowedPages = PAGE_ACCESS[currentRole] ?? PAGE_ACCESS[defaultRole];

  // Debug logging
  useEffect(() => {
    console.log("Auth state:", { isAuthenticated, isLoading, user });
  }, [isAuthenticated, isLoading, user]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setIsCommandPaletteOpen(true);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Validate current page is in allowed pages for this role
  useEffect(() => {
    if (!allowedPages.includes(currentPage)) {
      setCurrentPage(allowedPages[0]);
    }
  }, [allowedPages, currentPage]);

  console.log(
    "Rendering App - isAuthenticated:",
    isAuthenticated,
    "isLoading:",
    isLoading,
  );

  // Show loading state while checking authentication
  if (isLoading) {
    console.log("Showing loading state");
    return (
      <div
        className={`min-h-screen flex items-center justify-center ${
          theme === "dark"
            ? "bg-gradient-to-br from-[#0A1628] to-[#111827]"
            : "bg-gradient-to-br from-gray-50 to-gray-100"
        }`}
      >
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-primary/30 border-t-primary rounded-full animate-spin mx-auto mb-4" />
          <p className="text-foreground-secondary">Loading...</p>
        </div>
      </div>
    );
  }

  // Show auth page if not authenticated
  if (!isAuthenticated) {
    console.log("Showing auth page");
    return <AuthPage />;
  }

  console.log("Showing dashboard");

  const renderPage = () => {
    switch (currentPage) {
      case "dashboard":
        return <Dashboard />;
      case "forecast":
        return <ForecastPage />;
      case "data":
        return <DataPage />;
      case "analytics":
        return <AnalyticsPage />;
      case "settings":
        return <SettingsPage />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div
      className={`min-h-screen transition-colors duration-300 ${
        theme === "dark"
          ? "bg-gradient-to-br from-[#0A1628] to-[#111827] text-white"
          : "bg-gradient-to-br from-gray-50 to-gray-100 text-gray-900"
      }`}
    >
      {/* Animated background effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div
          className={`absolute top-0 right-0 w-[500px] h-[500px] rounded-full blur-[120px] animate-float ${
            theme === "dark" ? "bg-primary/10" : "bg-primary/5"
          }`}
        />
        <div
          className={`absolute bottom-0 left-0 w-[600px] h-[600px] rounded-full blur-[120px] animate-float ${
            theme === "dark" ? "bg-purple-500/10" : "bg-purple-500/5"
          }`}
          style={{ animationDelay: "1s" }}
        />
        <div
          className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] rounded-full blur-[100px] animate-float ${
            theme === "dark" ? "bg-cyan-500/5" : "bg-cyan-500/3"
          }`}
          style={{ animationDelay: "2s" }}
        />
      </div>

      <div className="relative z-10">
        <Header onPageChange={setCurrentPage} />
        <div className="flex">
          <Sidebar
            activePage={currentPage}
            onPageChange={setCurrentPage}
            allowedPages={allowedPages}
          />
          <main className="flex-1 ml-[280px] p-8">
            <AnimatePresence mode="wait">
              <motion.div
                key={currentPage}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                {renderPage()}
              </motion.div>
            </AnimatePresence>
          </main>
        </div>
      </div>

      {/* Command Palette */}
      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
        allowedPages={allowedPages}
        onNavigate={(page) => {
          setCurrentPage(page);
          setIsCommandPaletteOpen(false);
        }}
      />
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <DataProvider>
          <AppContent />
        </DataProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}
