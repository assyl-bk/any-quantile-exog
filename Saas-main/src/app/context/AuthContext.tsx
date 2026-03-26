import {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";

export type UserRole =
  | "energy_grid_operator"
  | "energy_trader"
  | "energy_planner"
  | "system_administrator";

interface User {
  id: number;
  email: string;
  name: string;
  role: UserRole;
  is_active?: boolean;
  created_at?: string;
  api_key?: string; // Full key from login/signup Token response
  profile_image?: string; // URL to profile image
}

interface RolesMatrix {
  roles: Record<UserRole, string[]>;
}

interface AuthContextType {
  user: User | null;
  capabilities: string[];
  rolesMatrix: RolesMatrix | null;
  login: (email: string, password: string) => Promise<void>;
  signup: (
    name: string,
    email: string,
    password: string,
    role: UserRole,
  ) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

function parseApiError(payload: unknown, fallbackMessage: string): string {
  if (!payload || typeof payload !== "object") {
    return fallbackMessage;
  }

  const response = payload as { detail?: unknown };
  if (typeof response.detail === "string") {
    return response.detail;
  }

  if (Array.isArray(response.detail)) {
    const firstDetail = response.detail[0] as { msg?: unknown } | undefined;
    if (firstDetail && typeof firstDetail.msg === "string") {
      return firstDetail.msg;
    }
    return fallbackMessage;
  }

  return fallbackMessage;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [capabilities, setCapabilities] = useState<string[]>([]);
  const [rolesMatrix, setRolesMatrix] = useState<RolesMatrix | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Check if user is already logged in on mount
  useEffect(() => {
    const token = localStorage.getItem("auth_token");
    if (token) {
      // Validate token and get user info
      validateToken(token);
    } else {
      setIsLoading(false);
    }
  }, []);

  const validateToken = async (token: string) => {
    try {
      const response = await fetch("http://localhost:8000/api/auth/validate", {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const userData = await response.json();
        console.log("User validated:", userData);
        // Rehydrate the full API key via dedicated endpoint
        // (GET /api/keys/ only returns masked prefixes, not the full key)
        let apiKey: string | undefined;
        try {
          const keyRes = await fetch(
            "http://localhost:8000/api/auth/my-api-key",
            {
              headers: { Authorization: `Bearer ${token}` },
            },
          );
          if (keyRes.ok) {
            const keyData = await keyRes.json();
            apiKey = keyData.api_key ?? undefined;
          }
        } catch {
          /* non-fatal */
        }
        setUser({ ...userData, api_key: apiKey });
        await Promise.all([fetchCapabilities(token), fetchRolesMatrix(token)]);
      } else {
        console.log("Token validation failed, clearing storage");
        localStorage.removeItem("auth_token");
        setCapabilities([]);
        setRolesMatrix(null);
      }
    } catch (error) {
      console.error("Token validation failed:", error);
      localStorage.removeItem("auth_token");
      setCapabilities([]);
      setRolesMatrix(null);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchCapabilities = async (token: string) => {
    try {
      const response = await fetch(
        "http://localhost:8000/api/auth/capabilities",
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        },
      );

      if (response.ok) {
        const data = await response.json();
        console.log("Capabilities fetched:", data);
        setCapabilities(data.capabilities || []);
      } else {
        console.warn("Failed to fetch capabilities:", response.status);
      }
    } catch (error) {
      console.error("Error fetching capabilities:", error);
    }
  };

  const fetchRolesMatrix = async (token: string) => {
    try {
      const response = await fetch(
        "http://localhost:8000/api/auth/roles-matrix",
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        },
      );

      if (response.ok) {
        const data = await response.json();
        console.log("RolesMatrix fetched:", data);
        setRolesMatrix(data);
      } else {
        console.warn("Failed to fetch roles matrix:", response.status);
      }
    } catch (error) {
      console.error("Error fetching roles matrix:", error);
    }
  };

  const login = async (email: string, password: string) => {
    try {
      const response = await fetch("http://localhost:8000/api/auth/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        const errorPayload = await response.json();
        throw new Error(parseApiError(errorPayload, "Login failed"));
      }

      const data = await response.json();
      console.log("Login successful:", data);
      localStorage.setItem("auth_token", data.access_token);
      setUser({ ...data.user, api_key: data.api_key ?? undefined });
      setIsLoading(false); // Mark loading as complete
      // Fetch capabilities and roles matrix, but don't block login on these
      await Promise.allSettled([
        fetchCapabilities(data.access_token),
        fetchRolesMatrix(data.access_token),
      ]);
      console.log("Login complete, user state updated");
    } catch (error) {
      console.error("Login error:", error);
      setIsLoading(false); // Stop loading even on error
      throw error;
    }
  };

  const signup = async (
    name: string,
    email: string,
    password: string,
    role: UserRole,
  ) => {
    try {
      const response = await fetch("http://localhost:8000/api/auth/signup", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ name, email, password, role }),
      });

      if (!response.ok) {
        const errorPayload = await response.json();
        throw new Error(parseApiError(errorPayload, "Signup failed"));
      }

      const data = await response.json();
      console.log("Signup successful:", data);
      // Don't auto-login after signup - user should login manually
      // This allows them to see the login page and proceed accordingly
    } catch (error) {
      console.error("Signup error:", error);
      throw error;
    }
  };

  const logout = () => {
    localStorage.removeItem("auth_token");
    setUser(null);
    setCapabilities([]);
    setRolesMatrix(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        capabilities,
        rolesMatrix,
        login,
        signup,
        logout,
        isAuthenticated: !!user,
        isLoading,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
