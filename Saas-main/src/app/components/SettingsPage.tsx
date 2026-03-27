import { useState, useEffect } from "react";
import { GlassCard } from "./GlassCard";
import {
  User,
  Key,
  Shield,
  Copy,
  Check,
  Eye,
  EyeOff,
  Upload,
} from "lucide-react";
import { motion } from "motion/react";
import { useAuth } from "../context/AuthContext";

const SETTINGS_TABS = [
  { id: "profile", label: "Profile", icon: User },
  { id: "api", label: "API Keys", icon: Key },
];

const ROLE_LABELS: Record<string, string> = {
  energy_grid_operator: "Energy Grid Operator",
  energy_trader: "Energy Trader",
  energy_planner: "Energy Planner",
  system_administrator: "System Administrator",
};

function Toggle({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-primary/50 ${
        checked ? "bg-primary" : "bg-white/20"
      }`}
    >
      <span
        className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow-lg ring-0 transition duration-200 ease-in-out ${
          checked ? "translate-x-5" : "translate-x-0"
        }`}
      />
    </button>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button
      onClick={handleCopy}
      className="p-1.5 rounded-lg hover:bg-white/10 transition-colors"
      title="Copy to clipboard"
    >
      {copied ? (
        <Check className="w-4 h-4 text-success" />
      ) : (
        <Copy className="w-4 h-4 text-foreground-secondary" />
      )}
    </button>
  );
}

// ── SettingsPage ──────────────────────────────────────────────────────────────

export function SettingsPage() {
  const { user } = useAuth();
  const token = localStorage.getItem("auth_token");

  const [activeTab, setActiveTab] = useState("profile");
  const [showApiKey, setShowApiKey] = useState(false);
  const [savedProfile, setSavedProfile] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [profileImage, setProfileImage] = useState<string | null>(
    user?.profile_image ?? null,
  );
  const [editedProfile, setEditedProfile] = useState({
    name: user?.name ?? "",
    email: user?.email ?? "",
  });

  // Notification toggles with threshold
  const defaultNotifications = {
    email: true,
    peak_demand: true,
    peak_demand_threshold: 8000, // Default threshold
    model_report: false,
    sys_updates: false,
  };

  const [notifications, setNotifications] = useState(defaultNotifications);
  const [savedNotifications, setSavedNotifications] =
    useState(defaultNotifications);

  // Check if there are unsaved changes
  const hasChanges =
    JSON.stringify(notifications) !== JSON.stringify(savedNotifications);

  // Load saved notification preferences on mount
  useEffect(() => {
    if (token && user?.id) {
      loadNotificationPreferences();
    }
  }, [user?.id, token]);

  const loadNotificationPreferences = async () => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/user/${user?.id}/notifications`,
        {
          headers: { Authorization: `Bearer ${token}` },
        },
      );
      if (response.ok) {
        const data = await response.json();
        setNotifications(data);
        setSavedNotifications(data); // Track the saved state
      }
    } catch (error) {
      console.error("Failed to load notification preferences:", error);
    }
  };

  const saveNotificationPreferences = async () => {
    if (!token) {
      console.error("❌ No auth token found");
      alert("Error: Not authenticated. Please log in again.");
      return;
    }

    if (!user?.id) {
      console.error("❌ No user ID");
      alert("Error: User ID not found.");
      return;
    }

    try {
      console.log("📤 Sending notifications:", notifications);
      console.log(
        "🔑 Token:",
        token ? token.substring(0, 20) + "..." : "MISSING",
      );
      console.log("👤 User ID:", user.id);

      const response = await fetch(
        `http://localhost:8000/api/user/${user.id}/notifications`,
        {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify(notifications),
        },
      );

      const responseText = await response.text();
      console.log("Response status:", response.status);
      console.log("Response text:", responseText);

      if (!response.ok) {
        throw new Error(`Failed to save: ${response.status} - ${responseText}`);
      }
      console.log("✅ Preferences saved successfully");
      setSavedNotifications(notifications); // Update saved state after success
    } catch (error) {
      console.error("Error saving notifications:", error);
      alert(`Error: ${error}`);
    }
  };

  const cancelNotificationChanges = () => {
    setNotifications(savedNotifications); // Revert to last saved state
  };

  const toggleNotif = (key: keyof typeof notifications) => {
    setNotifications((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleProfileImageUpload = async (file: File) => {
    if (!token) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch(
        `http://localhost:8000/api/user/${user?.id}/profile-image`,
        {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
          body: formData,
        },
      );

      if (response.ok) {
        const data = await response.json();
        setProfileImage(data.profile_image);
      } else {
        throw new Error("Failed to upload image");
      }
    } catch (error) {
      console.error("Profile image upload failed:", error);
    } finally {
      setIsUploading(false);
    }
  };

  const handleProfileImageSelect = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (file) {
      handleProfileImageUpload(file);
    }
  };

  const saveProfile = async () => {
    if (!token) return;

    try {
      const response = await fetch(
        `http://localhost:8000/api/user/${user?.id}/profile`,
        {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify(editedProfile),
        },
      );

      if (response.ok) {
        setSavedProfile(true);
        setTimeout(() => setSavedProfile(false), 2500);
      } else {
        throw new Error("Failed to save profile");
      }
    } catch (error) {
      console.error("Error saving profile:", error);
    }
  };

  // Derive initials from user name
  const initials = user?.name
    ? user.name
        .split(" ")
        .slice(0, 2)
        .map((n) => n[0])
        .join("")
        .toUpperCase()
    : "??";

  // Mask API key — show last 6 chars only when hidden
  const maskedKey = user?.api_key
    ? showApiKey
      ? user.api_key
      : `••••••••••••${user.api_key.slice(-6)}`
    : "No API key available";

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="grid grid-cols-[220px_1fr] gap-6">
      {/* Sidebar */}
      <div className="space-y-1">
        {SETTINGS_TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all text-sm ${
                isActive
                  ? "bg-primary/10 text-primary border border-primary/30"
                  : "bg-white/5 text-foreground-secondary hover:bg-white/10 border border-transparent"
              }`}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              <span className="font-medium">{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Content */}
      <div className="space-y-6 min-w-0">
        {/* ── PROFILE ── */}
        {activeTab === "profile" && (
          <>
            <div>
              <h2 className="text-2xl font-bold text-foreground mb-1">
                Profile Settings
              </h2>
              <p className="text-sm text-foreground-secondary">
                Manage your personal information and preferences
              </p>
            </div>

            {/* Avatar + name */}
            <GlassCard>
              <div className="flex items-start gap-6 mb-6">
                <div className="relative flex-shrink-0">
                  {profileImage ? (
                    <img
                      src={profileImage}
                      alt="Profile"
                      className="w-20 h-20 rounded-2xl object-cover"
                    />
                  ) : (
                    <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary to-purple-500 flex items-center justify-center text-white text-2xl font-bold">
                      {initials}
                    </div>
                  )}
                  <label className="absolute bottom-0 right-0 bg-primary text-white p-2 rounded-full cursor-pointer hover:bg-primary/80 transition-colors">
                    <Upload className="w-4 h-4" />
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleProfileImageSelect}
                      disabled={isUploading}
                      className="hidden"
                    />
                  </label>
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-xl font-bold text-foreground truncate">
                    {user?.name ?? "—"}
                  </h3>
                  <p className="text-sm text-foreground-secondary mb-1">
                    {ROLE_LABELS[user?.role ?? ""] ??
                      user?.role ??
                      "Unknown role"}
                  </p>
                  <p className="text-xs text-muted-foreground font-mono truncate">
                    {user?.email ?? "No email on record"}
                  </p>
                </div>
              </div>

              {/* Fields */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-foreground-secondary mb-1.5 block">
                    Name
                  </label>
                  <input
                    type="text"
                    value={editedProfile.name}
                    onChange={(e) =>
                      setEditedProfile((prev) => ({
                        ...prev,
                        name: e.target.value,
                      }))
                    }
                    className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2.5 text-white text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-foreground-secondary mb-1.5 block">
                    Email Address
                  </label>
                  <input
                    type="email"
                    value={editedProfile.email}
                    onChange={(e) =>
                      setEditedProfile((prev) => ({
                        ...prev,
                        email: e.target.value,
                      }))
                    }
                    className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2.5 text-white text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-foreground-secondary mb-1.5 block">
                    Role
                  </label>
                  <input
                    type="text"
                    readOnly
                    value={ROLE_LABELS[user?.role ?? ""] ?? user?.role ?? "—"}
                    className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2.5 text-foreground-secondary text-sm cursor-not-allowed"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-3 mt-6 pt-6 border-t border-white/10">
                <button className="px-5 py-2.5 rounded-lg bg-white/5 text-foreground-secondary border border-white/10 text-sm font-medium hover:bg-white/10 transition-colors">
                  Cancel
                </button>
                <button
                  onClick={saveProfile}
                  className="px-5 py-2.5 rounded-lg bg-gradient-to-r from-primary to-[#2563EB] text-white text-sm font-medium hover:shadow-lg hover:shadow-primary/30 transition-all flex items-center gap-2"
                >
                  {savedProfile ? (
                    <>
                      <Check className="w-4 h-4" /> Saved!
                    </>
                  ) : (
                    "Save Changes"
                  )}
                </button>
              </div>
            </GlassCard>

            {/* Notification Preferences */}
            <GlassCard>
              <h3 className="text-lg font-bold text-foreground mb-4">
                Notification Preferences
              </h3>
              <div className="divide-y divide-white/10">
                <div className="flex items-center justify-between py-4 first:pt-0 last:pb-0">
                  <div>
                    <p className="text-sm font-medium text-foreground">
                      Email Notifications
                    </p>
                    <p className="text-xs text-foreground-secondary mt-0.5">
                      Receive forecast alerts via email
                    </p>
                  </div>
                  <Toggle
                    checked={notifications.email}
                    onChange={() => toggleNotif("email")}
                  />
                </div>

                <div className="py-4">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <p className="text-sm font-medium text-foreground">
                        Peak Demand Alerts
                      </p>
                      <p className="text-xs text-foreground-secondary mt-0.5">
                        Get notified when demand exceeds threshold
                      </p>
                    </div>
                    <Toggle
                      checked={notifications.peak_demand}
                      onChange={() => toggleNotif("peak_demand")}
                    />
                  </div>
                  {notifications.peak_demand && (
                    <div className="ml-0 bg-white/5 rounded-lg p-3 border border-white/10">
                      <label className="text-xs font-medium text-foreground-secondary block mb-2">
                        Demand Threshold (MW)
                      </label>
                      <input
                        type="number"
                        value={notifications.peak_demand_threshold}
                        onChange={(e) =>
                          setNotifications((prev) => ({
                            ...prev,
                            peak_demand_threshold:
                              parseInt(e.target.value) || 0,
                          }))
                        }
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                        min="0"
                      />
                      <p className="text-xs text-muted-foreground mt-2">
                        You'll receive notifications when demand exceeds{" "}
                        <span className="text-primary font-semibold">
                          {notifications.peak_demand_threshold} MW
                        </span>
                      </p>
                    </div>
                  )}
                </div>

                <div className="flex items-center justify-between py-4 first:pt-0 last:pb-0">
                  <div>
                    <p className="text-sm font-medium text-foreground">
                      Model Performance Reports
                    </p>
                    <p className="text-xs text-foreground-secondary mt-0.5">
                      Weekly model accuracy summaries
                    </p>
                  </div>
                  <Toggle
                    checked={notifications.model_report}
                    onChange={() => toggleNotif("model_report")}
                  />
                </div>

                <div className="flex items-center justify-between py-4 first:pt-0 last:pb-0">
                  <div>
                    <p className="text-sm font-medium text-foreground">
                      System Updates
                    </p>
                    <p className="text-xs text-foreground-secondary mt-0.5">
                      Platform updates and maintenance notices
                    </p>
                  </div>
                  <Toggle
                    checked={notifications.sys_updates}
                    onChange={() => toggleNotif("sys_updates")}
                  />
                </div>
              </div>

              <div
                className={`flex justify-end gap-3 mt-6 pt-6 border-t border-white/10 ${!hasChanges && "hidden"}`}
              >
                <button
                  onClick={cancelNotificationChanges}
                  className="px-5 py-2.5 rounded-lg bg-white/5 text-foreground-secondary border border-white/10 text-sm font-medium hover:bg-white/10 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={saveNotificationPreferences}
                  className="px-5 py-2.5 rounded-lg bg-gradient-to-r from-primary to-[#2563EB] text-white text-sm font-medium hover:shadow-lg hover:shadow-primary/30 transition-all"
                >
                  Save Preferences
                </button>
              </div>
            </GlassCard>
          </>
        )}

        {/* ── API KEYS ── */}
        {activeTab === "api" && (
          <>
            <div>
              <h2 className="text-2xl font-bold text-foreground mb-1">
                API Keys
              </h2>
              <p className="text-sm text-foreground-secondary">
                Your key for programmatic access to the forecast API
              </p>
            </div>

            {/* Security notice */}
            <GlassCard className="bg-gradient-to-br from-warning/5 to-red-500/5 border-warning/20">
              <div className="flex items-start gap-3">
                <Shield className="w-5 h-5 text-warning flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-foreground mb-0.5">
                    Keep your API key secure
                  </p>
                  <p className="text-sm text-foreground-secondary">
                    Never share your key in public repositories or client-side
                    code. Use environment variables instead.
                  </p>
                </div>
              </div>
            </GlassCard>

            {/* Real API key from auth context */}
            <GlassCard>
              <div className="flex items-start justify-between mb-4">
                <div className="min-w-0 flex-1">
                  <h3 className="text-sm font-semibold text-foreground mb-1">
                    Your API Key
                  </h3>
                  <div className="flex items-center gap-2">
                    <code className="text-sm font-mono text-foreground-secondary bg-white/5 px-3 py-1.5 rounded-lg border border-white/10 flex-1 min-w-0 truncate">
                      {maskedKey}
                    </code>
                    {user?.api_key && <CopyButton text={user.api_key} />}
                  </div>
                </div>
                <button
                  onClick={() => setShowApiKey((v) => !v)}
                  className="ml-4 flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-foreground-secondary text-sm font-medium transition-colors border border-white/10 flex-shrink-0"
                >
                  {showApiKey ? (
                    <>
                      <EyeOff className="w-4 h-4" /> Hide
                    </>
                  ) : (
                    <>
                      <Eye className="w-4 h-4" /> Reveal
                    </>
                  )}
                </button>
              </div>

              <div className="flex items-center justify-between text-xs text-foreground-secondary pt-4 border-t border-white/10">
                <div className="flex gap-4">
                  <span>
                    Status:{" "}
                    <span className="text-success font-medium">Active</span>
                  </span>
                  <span>
                    User ID:{" "}
                    <span className="font-mono text-foreground">
                      {user?.id ?? "—"}
                    </span>
                  </span>
                </div>
                <span className="text-muted-foreground italic">
                  Contact an admin to rotate this key
                </span>
              </div>
            </GlassCard>

            {/* Usage example */}
            <GlassCard>
              <h3 className="text-sm font-semibold text-foreground mb-3">
                Usage Example
              </h3>
              <pre className="text-xs font-mono text-foreground-secondary bg-black/30 rounded-lg p-4 overflow-x-auto border border-white/10">
                {`POST /api/forecast/forecast
Content-Type: application/json
X-API-Key: ${user?.api_key ? maskedKey : "<your-api-key>"}

{
  "historical_data": [8234, 8100, 7980, ...],
  "quantiles": [0.05, 0.25, 0.5, 0.75, 0.9],
  "apply_cqr": false
}`}
              </pre>
            </GlassCard>
          </>
        )}
      </div>
    </div>
  );
}
