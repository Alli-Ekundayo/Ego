import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Brain,
  CheckCircle,
  Cpu,
  Database,
  FileCode,
  Sparkle,
  Globe,
  Sun,
  Moon,
} from "@phosphor-icons/react";
import { useUsers } from "./hooks/useUsers";

let globalTheme = "dark";
if (typeof window !== "undefined") {
  globalTheme = localStorage.getItem("theme") || "dark";
  if (globalTheme === "dark") {
    window.document.documentElement.classList.add("dark");
  } else {
    window.document.documentElement.classList.remove("dark");
  }
}

const themeListeners = new Set();

export function useTheme() {
  const [theme, setThemeState] = useState(globalTheme);

  useEffect(() => {
    const listener = (newTheme) => setThemeState(newTheme);
    themeListeners.add(listener);
    return () => themeListeners.delete(listener);
  }, []);

  const setTheme = (newTheme) => {
    globalTheme = newTheme;
    const root = window.document.documentElement;
    if (newTheme === "dark") {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }
    localStorage.setItem("theme", newTheme);
    themeListeners.forEach((l) => l(newTheme));
  };

  const toggleTheme = () => setTheme(theme === "dark" ? "light" : "dark");

  return [theme, toggleTheme];
}

import ResultsGrid from "./components/ResultsGrid";
import ReviewForm from "./components/ReviewForm";
import RecommendForm from "./components/RecommendForm";
import MagneticButton from "./components/MagneticButton";
import PulseDot from "./components/PulseDot";

const PAPER_LINKS = {
  taskA: "https://drive.google.com/file/d/1TSOcNd7uMc-C7fFba6vp3PGlZDTilBP2/view?usp=drive_link",
  taskB: "https://drive.google.com/file/d/12__mHA2at_Q0Yo82Ug8GPiNzPNtDApDF/view?usp=drive_link",
};

const NAV_ITEMS = [
  { label: "Home", href: "/", icon: Globe },
  { label: "Task A", href: "/task-a", icon: CheckCircle },
  { label: "Task B", href: "/task-b", icon: Sparkle },
];

function normalizePath(pathname) {
  if (!pathname || pathname === "/") return "/";
  return pathname.replace(/\/+$/, "");
}

function usePathname() {
  const [pathname, setPathname] = useState(() =>
    normalizePath(window.location.pathname),
  );

  useEffect(() => {
    const handlePopState = () => {
      setPathname(normalizePath(window.location.pathname));
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  return pathname;
}

function navigate(to) {
  const nextPath = normalizePath(to);
  if (nextPath === normalizePath(window.location.pathname)) return;
  window.history.pushState({}, "", nextPath);
  window.dispatchEvent(new PopStateEvent("popstate"));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function isModifiedEvent(event) {
  return event.metaKey || event.altKey || event.ctrlKey || event.shiftKey || event.button !== 0;
}

function AppLink({
  href,
  className = "",
  activeClassName = "",
  children,
  onClick,
  target,
  rel,
  ...rest
}) {
  const pathname = usePathname();
  const isExternal = /^https?:\/\//.test(href);
  const isActive = !isExternal && normalizePath(pathname) === normalizePath(href);

  return (
    <a
      href={href}
      target={target}
      rel={rel}
      onClick={(event) => {
        onClick?.(event);
        if (event.defaultPrevented) return;
        if (isExternal || target === "_blank" || isModifiedEvent(event)) return;
        event.preventDefault();
        navigate(href);
      }}
      className={`${className} ${isActive ? activeClassName : ""}`}
      {...rest}
    >
      {children}
    </a>
  );
}

function Section({ id, className = "", delay = 0, children, ...rest }) {
  return (
    <motion.section
      id={id}
      initial={{ opacity: 0, y: 14 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.18 }}
      transition={{ type: "spring", stiffness: 90, damping: 18, delay }}
      className={className}
      {...rest}
    >
      {children}
    </motion.section>
  );
}

function formatCategory(value) {
  return value
    ? value
      .split(/\s+/)
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ")
    : "—";
}

function buildPersonaSummary(user) {
  if (!user) return "—";

  const category = formatCategory(user.top_category).toLowerCase();
  const reviews = Number(user.review_count || 0);
  const rating = Number(user.mean_rating || 0);

  if (!category || category === "—") return "General shopper";
  if (reviews >= 100 && rating >= 4.2) return `${category} power reviewer`;
  if (reviews >= 50) return `${category} regular`;
  if (rating >= 4.2) return `${category} quality seeker`;
  return `${category} explorer`;
}

function UserDashboard({ title, users, loading, selectedUserId, onChange, selectedUser, color = "emerald", onGuestChange, guestUser }) {
  const [isGuest, setIsGuest] = useState(false);
  const [guestName, setGuestName] = useState("");
  const [guestPersona, setGuestPersona] = useState("");

  const colorMap = {
    emerald: {
      badgeBorder: "border-emerald-200/70 dark:border-emerald-900/50",
      badgeBg: "bg-emerald-50/80 dark:bg-emerald-950/40",
      badgeText: "text-emerald-700 dark:text-emerald-400",
      statText: "text-emerald-600 dark:text-emerald-450",
      toggleActive: "bg-emerald-600 dark:bg-emerald-500",
    },
    orange: {
      badgeBorder: "border-orange-200/70 dark:border-orange-900/50",
      badgeBg: "bg-orange-50/80 dark:bg-orange-950/40",
      badgeText: "text-orange-700 dark:text-orange-400",
      statText: "text-orange-600 dark:text-orange-450",
      toggleActive: "bg-orange-600 dark:bg-orange-500",
    }
  };
  const themeStyles = colorMap[color] || colorMap.emerald;

  const handleGuestToggle = (enabled) => {
    setIsGuest(enabled);
    if (!enabled) {
      onGuestChange?.(null);
    } else {
      // If already have a name, push it immediately
      if (guestName.trim()) {
        const slug = guestName.trim().toLowerCase().replace(/\s+/g, "_");
        onGuestChange?.({ user_id: `guest_${slug}`, name: guestName.trim(), persona: guestPersona.trim() });
      }
    }
  };

  const handleGuestInput = (name, persona) => {
    setGuestName(name);
    setGuestPersona(persona);
    if (name.trim()) {
      const slug = name.trim().toLowerCase().replace(/\s+/g, "_");
      onGuestChange?.({ user_id: `guest_${slug}`, name: name.trim(), persona: persona.trim() });
    } else {
      onGuestChange?.(null);
    }
  };

  const activeUser = isGuest ? guestUser : selectedUser;
  const stats = [
    {
      label: "Past reviews",
      value: isGuest ? "0" : (selectedUser ? selectedUser.review_count : "—"),
      note: isGuest ? "New user — no history yet" : "Count of historical reviews",
    },
    {
      label: "Average rating",
      value: isGuest ? "—" : (selectedUser ? selectedUser.mean_rating.toFixed(2) : "—"),
      note: isGuest ? "Cold-start mode active" : "Mean rating from the profile",
    },
    {
      label: "Top category",
      value: isGuest ? "—" : (selectedUser ? formatCategory(selectedUser.top_category) : "—"),
      note: isGuest ? "Inferred from persona on submit" : "Most frequent category",
    },
    {
      label: "Persona summary",
      value: isGuest ? (guestUser ? "New user" : "—") : (selectedUser ? buildPersonaSummary(selectedUser) : "—"),
      note: isGuest ? "Powered by cold-start retrieval" : "Derived from stats, not the raw category",
    },
  ];

  return (
    <section className="mx-auto max-w-[85rem] px-4 pb-20 sm:px-6 lg:px-12 relative z-10">
      <div className="rounded-2xl border border-zinc-200/70 bg-white/80 dark:border-zinc-800/80 dark:bg-zinc-900/80 p-6 shadow-[0_24px_50px_-36px_rgba(15,23,42,0.35)] dark:shadow-[0_24px_50px_-36px_rgba(0,0,0,0.7)] backdrop-blur">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
          <div className="max-w-2xl">
            <div className="flex items-center gap-3">
              <div className={`inline-flex items-center gap-2 rounded-full border ${themeStyles.badgeBorder} ${themeStyles.badgeBg} px-3 py-1 text-[10px] uppercase tracking-[0.22em] ${themeStyles.badgeText}`}>
                <PulseDot color={color} />
                User dashboard
              </div>
              {/* New user toggle */}
              <button
                type="button"
                onClick={() => handleGuestToggle(!isGuest)}
                className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-[10px] uppercase tracking-[0.18em] font-medium transition-all ${
                  isGuest
                    ? "border-zinc-950/20 dark:border-zinc-50/20 bg-zinc-950 dark:bg-zinc-50 text-white dark:text-zinc-950"
                    : "border-zinc-200/70 dark:border-zinc-700 bg-white dark:bg-zinc-900 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-50 dark:hover:bg-zinc-800"
                }`}
              >
                <span className={`h-1.5 w-1.5 rounded-full transition-colors ${isGuest ? "bg-white/80 dark:bg-zinc-950/80" : "bg-zinc-400 dark:bg-zinc-500"}`} />
                {isGuest ? "New user" : "New user?"}
              </button>
            </div>
            <h2 className="mt-4 text-3xl font-semibold tracking-tight text-zinc-950 dark:text-white">{title}</h2>
            <p className="mt-3 text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
              {isGuest
                ? "Enter a name and a short persona description. The system will use cold-start retrieval to generate personalised results."
                : <>Profiles are loaded from <code className="rounded bg-zinc-100 dark:bg-zinc-800 px-2 py-0.5 font-mono text-[0.85em] text-zinc-800 dark:text-zinc-200">/api/users</code>. The selected user feeds the endpoint payload below.</>}
            </p>
          </div>

          <div className="w-full max-w-md">
            {isGuest ? (
              <div className="flex flex-col gap-3">
                <label className="flex flex-col gap-1.5">
                  <span className="input-label">Your name</span>
                  <input
                    type="text"
                    placeholder="e.g. Amara Okonkwo"
                    value={guestName}
                    onChange={(e) => handleGuestInput(e.target.value, guestPersona)}
                    className="input-base w-full"
                    aria-label="New user name"
                  />
                </label>
                <label className="flex flex-col gap-1.5">
                  <span className="input-label">Persona description</span>
                  <textarea
                    placeholder="e.g. budget-conscious student who loves tech deals and fashion"
                    value={guestPersona}
                    onChange={(e) => handleGuestInput(guestName, e.target.value)}
                    rows={3}
                    className="input-base w-full resize-none"
                    aria-label="Guest persona description"
                  />
                  <span className="text-[11px] text-zinc-500 dark:text-zinc-450">Describe your shopping style to guide cold-start recommendations.</span>
                </label>
              </div>
            ) : (
              <label className="flex flex-col gap-2">
                <span className="input-label">User profile</span>
                <select
                  value={selectedUserId}
                  onChange={(e) => onChange(e.target.value)}
                  className="input-base w-full"
                  aria-label="Select a user profile"
                >
                  <option value="" disabled>
                    {loading ? "Loading users..." : "Select a user"}
                  </option>
                  {users.map((user) => (
                    <option key={user.user_id} value={user.user_id}>
                      {user.name}
                    </option>
                  ))}
                </select>
                <span className="text-xs text-zinc-500 dark:text-zinc-450">
                  Selected: <span className="font-medium text-zinc-900 dark:text-white">{selectedUser?.name || "—"}</span>
                </span>
                <span className="text-[11px] text-zinc-500 dark:text-zinc-450">Switching profiles updates both Task A and Task B requests.</span>
              </label>
            )}
          </div>
        </div>

        <div className="mt-8 grid gap-4 sm:grid-cols-2">
          {stats.map((stat) => (
            <article
              key={stat.label}
              className="rounded-xl border border-zinc-200/70 dark:border-zinc-800/70 bg-zinc-50/60 dark:bg-zinc-950/60 p-4"
            >
              <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">{stat.label}</p>
              <p className={`mt-2 text-2xl font-semibold ${themeStyles.statText}`}>
                <span className="font-mono">{stat.value}</span>
              </p>
              <p className="mt-2 text-xs leading-relaxed text-zinc-500 dark:text-zinc-450">{stat.note}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

function Shell({ footerText, footerLinks, footerMeta, children }) {
  const [theme, toggleTheme] = useTheme();

  return (
    <div className="min-h-[100dvh] bg-zinc-50 text-zinc-950 antialiased dark:bg-zinc-950 dark:text-zinc-50 transition-colors duration-300 relative">
      <a
        href="#content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-40 focus:rounded-full focus:bg-white dark:focus:bg-zinc-900 focus:px-4 focus:py-2 focus:text-sm focus:text-zinc-950 dark:focus:text-zinc-50 focus:shadow-lg focus:outline-none"
        aria-label="Skip to main content"
      >
        Skip to content
      </a>

      <nav className="fixed left-1/2 top-4 z-40 mx-auto flex h-16 w-[min(100%-2rem,85rem)] -translate-x-1/2 items-center justify-between px-4 sm:px-6 lg:px-10 transition-colors duration-305">
        <div className="flex items-center gap-3 font-medium tracking-tight">
          <AppLink href="/" className="text-lg font-semibold text-zinc-950 dark:text-white transition-colors hover:text-zinc-900 dark:hover:text-zinc-200">
            Ego
          </AppLink>
          <span className="text-zinc-300 dark:text-zinc-700 font-light text-xs">/</span>
          <div className="hidden items-center gap-2 sm:flex">
            <PulseDot />
            <span className="text-[11px] text-zinc-500 dark:text-zinc-400 font-medium">FastAPI • LangGraph • Turbovec</span>
          </div>
        </div>

        <div className="hidden items-center gap-4 md:flex">
          <div className="flex items-center gap-2">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              return (
                <AppLink
                  key={item.href}
                  href={item.href}
                  className="group relative flex items-center gap-2 rounded-full border border-transparent px-4 py-2 text-sm text-zinc-650 dark:text-zinc-400 transition-all hover:border-zinc-200/70 dark:hover:border-zinc-800 hover:bg-white dark:hover:bg-zinc-850 hover:text-zinc-950 dark:hover:text-zinc-100 hover:shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/40"
                  activeClassName="border-zinc-200/70 dark:border-zinc-800/80 bg-white dark:bg-zinc-800 text-zinc-950 dark:text-white shadow-sm"
                >
                  {Icon && <Icon size={14} weight="bold" />}
                  <span>{item.label}</span>
                  <span className="absolute -bottom-1 left-1/2 h-0.5 w-0 -translate-x-1/2 rounded-full bg-emerald-500/70 transition-all duration-300 group-hover:w-full" />
                </AppLink>
              );
            })}
          </div>
          <button
            onClick={toggleTheme}
            type="button"
            className="flex items-center justify-center rounded-full border border-zinc-200/70 bg-white dark:border-zinc-800 dark:bg-zinc-900 p-2.5 text-zinc-600 dark:text-zinc-450 hover:bg-zinc-50 dark:hover:bg-zinc-800 hover:text-zinc-950 dark:hover:text-white transition-all focus:outline-none focus:ring-2 focus:ring-emerald-500/40 cursor-pointer shadow-sm"
            aria-label="Toggle Dark Mode"
          >
            {theme === "dark" ? <Sun size={15} weight="bold" /> : <Moon size={15} weight="bold" />}
          </button>
        </div>

        <div className="flex items-center gap-2 md:hidden">
          {NAV_ITEMS.map((item) => (
            <AppLink
              key={item.href}
              href={item.href}
              className="rounded-full border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-3 py-2 text-[11px] font-medium text-zinc-600 dark:text-zinc-400 transition"
              activeClassName="bg-white dark:bg-zinc-800 text-zinc-950 dark:text-white border-zinc-300/70 dark:border-zinc-700 shadow-sm"
            >
              {item.label}
            </AppLink>
          ))}
          <button
            onClick={toggleTheme}
            type="button"
            className="flex items-center justify-center rounded-full border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-2.5 text-zinc-650 dark:text-zinc-400 transition cursor-pointer shadow-sm"
            aria-label="Toggle Dark Mode"
          >
            {theme === "dark" ? <Sun size={12} weight="bold" /> : <Moon size={12} weight="bold" />}
          </button>
        </div>
      </nav>

      <main
        id="content"
        className="relative min-h-[100dvh] overflow-hidden pt-24 text-zinc-950 dark:text-zinc-50 selection:bg-emerald-100 selection:text-emerald-900 dark:selection:bg-emerald-900 dark:selection:text-emerald-100 z-10"
      >
        <div className="pointer-events-none absolute inset-0 z-0">
          <div className="absolute inset-0 bg-grid opacity-60 dark:opacity-85" />
          <div className="absolute -top-72 left-1/2 h-[620px] w-[620px] -translate-x-1/2 rounded-full bg-[radial-gradient(circle,rgba(16,185,129,0.12),transparent_60%)] blur-[140px]" />
          <div className="absolute bottom-[-280px] right-[-140px] h-[560px] w-[560px] rounded-full bg-[radial-gradient(circle,rgba(16,185,129,0.08),transparent_60%)] blur-[160px]" />
        </div>

        {children}

        <footer className="mt-24 relative z-10">
          <div className="mx-auto max-w-[85rem] px-4 py-16 sm:px-6 lg:px-12">
            <div className="grid grid-cols-1 gap-12 pb-14 md:grid-cols-2">
              <div className="max-w-md">
                <div className="mb-5 flex items-center gap-2">
                  <span className="text-lg font-semibold text-zinc-950 dark:text-white">Ego</span>
                  <span className="text-zinc-300 dark:text-zinc-700 font-light text-xs">/</span>
                  <span className="text-xs text-zinc-500 dark:text-zinc-400">recommendation system</span>
                </div>
                <p className="text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">{footerText}</p>
              </div>

              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                {footerLinks.map((link) => (
                  <AppLink
                    key={link.label}
                    href={link.href}
                    target={link.external ? "_blank" : undefined}
                    rel={link.external ? "noreferrer" : undefined}
                    className="group rounded-xl border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-4 py-3 text-xs font-medium text-zinc-600 dark:text-zinc-400 transition-all hover:bg-zinc-50 dark:hover:bg-zinc-800 hover:text-zinc-950 dark:hover:text-white hover:-translate-y-0.5 hover:border-zinc-300/70 dark:hover:border-zinc-700 hover:shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/40"
                  >
                    {link.label}
                  </AppLink>
                ))}
              </div>
            </div>

            <div className="flex flex-col gap-3 pt-10 text-xs text-zinc-500 dark:text-zinc-400 sm:flex-row sm:items-center sm:justify-between">
              <p>{footerMeta}</p>
              <div className="flex items-center gap-4 font-mono text-[11px]">
                <span className="flex items-center gap-1.5">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-500/80" />
                  FastAPI
                </span>
                <span>•</span>
                <span>LangGraph</span>
                <span>•</span>
                <span>Turbovec</span>
              </div>
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
}


function HomePage() {
  const taskCards = [
    {
      id: "task-a",
      eyebrow: "Task A",
      title: "User Modelling",
      icon: CheckCircle,
      description:
        "Build an agent that understands users deploy enough to simulate thier reviews, capturing tone, rating behavior and contextual nuances.",
      bullets: [
        "Simulate star ratings and written reviews for unseen items",
        "Leverage user history, item metadata, and contextual signals",
        "Evaluate on review quality, rating accuracy and behavioral fidelity",
      ],
      href: "/task-a",
      styles: {
        badgeBorder: "border-emerald-200/70 dark:border-emerald-900/50",
        badgeBg: "bg-emerald-50 dark:bg-emerald-950/40",
        badgeText: "text-emerald-700 dark:text-emerald-400",
        bulletBg: "bg-emerald-500/70",
        linkHover: "hover:text-emerald-600 dark:hover:text-emerald-450",
        ringFocus: "focus:ring-emerald-500/40",
      }
    },
    {
      id: "task-b",
      eyebrow: "Task B",
      title: "Contextual Recommendations",
      icon: Sparkle,
      description:
        "Build an agent that delivers personalises reccomendations, going beyond collaborative filtering to contextual, conversation retrival.",
      bullets: [
        "Rank and recomend items tailored to individual user context",
        "Handle cold-start, cross domain and multiturn scenerios",
        "Design agentic workflow that reason before recomending",
      ],
      href: "/task-b",
      styles: {
        badgeBorder: "border-orange-200/70 dark:border-orange-900/50",
        badgeBg: "bg-orange-50 dark:bg-orange-950/40",
        badgeText: "text-orange-700 dark:text-orange-400",
        bulletBg: "bg-orange-500/70",
        linkHover: "hover:text-orange-600 dark:hover:text-orange-450",
        ringFocus: "focus:ring-orange-500/40",
      }
    },
  ];

  const routes = ["POST /simulate-review", "POST /recommend", "GET /users", "GET /products"];

  const timeline = [
    {
      label: "Source Reviews",
      title: "Jumia review data",
      body:
        "The repository uses real Jumia product reviews scraped from the Nigerian market as the language source for culturally grounded recommendations and review generation.",
    },
    {
      label: "Profile Store",
      title: "data/user_profiles.json",
      body:
        "User profiles are aggregated into a shared profile store with review counts, category preferences, mean ratings, and cached training reviews.",
    },
    {
      label: "Indexing + Cache",
      title: "data/items.json - Turbovec - diskcache",
      body:
        "Products are loaded for catalogue browsing, embedded into Turbovec, and warmed alongside the BM25 corpus, SQLite response cache, and disk-backed embedding cache.",
    },
  ];

  const architecture = [
    { icon: Cpu, label: "FastAPI Gateway", value: "api/main.py", detail: "Bootstrap, lifecycle, and routing." },
    { icon: Brain, label: "LangGraph Pipelines", value: "graphs/task_a.py + task_b.py", detail: "Task graphs and node orchestration." },
    { icon: Database, label: "Hybrid Retrieval", value: "Turbovec + BM25 + RRF", detail: "Dense and sparse fusion." },
    { icon: Sparkle, label: "Model Stack", value: "qwen-plus + all-MiniLM-L6-v2", detail: "LLM and embeddings." },
  ];

  const modules = [
    {
      name: "api/main.py",
      role: "FastAPI app, startup preloading, /health, /simulate-review, /recommend, /users, /products",
    },
    {
      name: "graphs/task_a.py",
      role: "User modelling graph with profile retrieval, rating prediction, style analysis, and review generation",
    },
    {
      name: "graphs/task_b.py",
      role: "Recommendation graph with context extraction, cold-start handling, hybrid retrieval, and multiturn refinement",
    },
  ];

  return (
    <Shell
      footerText="Built for transparent recommendation workflows on real Jumia review data, with separate user modelling and contextual recommendation graphs."
      footerLinks={[
        { label: "Task A page", href: "/task-a" },
        { label: "Task B page", href: "/task-b" },
        { label: "Task A paper", href: PAPER_LINKS.taskA, external: true },
        { label: "Task B paper", href: PAPER_LINKS.taskB, external: true },
      ]}
      footerMeta="Data: data/items.json - data/user_profiles.json - data/jumia_reviews.json"
    >
      <Section id="top" className="relative mx-auto max-w-[85rem] px-4 pb-24 pt-32 sm:px-6 lg:px-12 relative z-10">
        <div className="grid gap-12 lg:grid-cols-[1.15fr_0.85fr]">
          <div className="relative">
            <div className="inline-flex items-center gap-2 rounded-full border border-emerald-200/70 dark:border-emerald-900/50 bg-emerald-50/80 dark:bg-emerald-950/40 px-4 py-1.5 text-[10px] uppercase tracking-[0.22em] text-emerald-700 dark:text-emerald-450">
              <PulseDot />
              Jumia Nigeria recommendation studio
            </div>
            <h1 className="mt-8 text-4xl font-semibold tracking-tighter leading-none text-zinc-950 dark:text-white md:text-6xl">
              Ego maps <span className="text-emerald-600 dark:text-emerald-450 italic font-display text-[1.1em]">shopper intent</span> to grounded results.
            </h1>
            <p className="mt-6 max-w-[60ch] text-base leading-relaxed text-zinc-600 dark:text-zinc-400">
              Run Task A to simulate culturally aligned reviews. Run Task B to return ranked
              recommendations from live context, session history, and domain filters.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <MagneticButton
                type="button"
                className="group inline-flex items-center gap-2 rounded-xl bg-zinc-950 dark:bg-zinc-50 px-6 py-3.5 text-sm font-semibold text-white dark:text-zinc-950 shadow-[0_10px_26px_-16px_rgba(9,9,11,0.6)] dark:shadow-[0_10px_26px_-16px_rgba(255,255,255,0.15)] transition-transform hover:-translate-y-0.5 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:ring-offset-2"
                onClick={() => navigate("/task-a")}
              >
                Open Task A
                <ArrowRight size={16} weight="bold" className="transition-transform group-hover:translate-x-1" />
              </MagneticButton>
              <AppLink
                className="inline-flex items-center gap-2 rounded-xl border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-5 py-3.5 text-sm font-semibold text-zinc-700 dark:text-zinc-300 transition-all hover:-translate-y-0.5 hover:border-zinc-300/70 dark:hover:border-zinc-700 hover:text-zinc-950 dark:hover:text-white focus:outline-none focus:ring-2 focus:ring-orange-500/40 focus:ring-offset-2"
                href="/task-b"
              >
                Open Task B
                <ArrowRight size={14} weight="bold" className="text-orange-600" />
              </AppLink>
            </div>
            <div className="mt-10 flex flex-wrap gap-6 text-xs text-zinc-500 dark:text-zinc-450">
              <div className="flex items-center gap-2">
                <span className="h-1 w-6 rounded-full bg-emerald-500/30" />
                <span>Live Jumia review corpus</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="h-1 w-6 rounded-full bg-emerald-500/30" />
                <span>Task-specific LangGraph nodes</span>
              </div>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <article className="rounded-2xl border border-zinc-200/70 dark:border-zinc-800/80 bg-white/90 dark:bg-zinc-900/90 p-6 shadow-[0_20px_40px_-28px_rgba(15,23,42,0.25)] dark:shadow-[0_20px_40px_-28px_rgba(0,0,0,0.5)] sm:col-span-2">
              <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">Model stack</p>
              <h3 className="mt-2 text-xl font-semibold text-zinc-950 dark:text-white">
                qwen-plus + all-MiniLM-L6-v2
              </h3>
              <p className="mt-3 text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
                LLM reasoning is paired with a compact embedding model for retrieval, similarity,
                and reranking.
              </p>
              <div className="mt-4 inline-flex items-center gap-2 rounded-full border border-emerald-200/70 dark:border-emerald-900/50 bg-emerald-50 dark:bg-emerald-950/40 px-3 py-1 text-[11px] font-medium text-emerald-700 dark:text-emerald-400">
                <PulseDot />
                Active in both tasks
              </div>
            </article>

            <article className="rounded-2xl border border-zinc-200/70 dark:border-zinc-800/80 bg-white/90 dark:bg-zinc-900/90 p-5 shadow-[0_16px_36px_-26px_rgba(15,23,42,0.25)] dark:shadow-[0_16px_36px_-26px_rgba(0,0,0,0.4)]">
              <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">Cache path</p>
              <p className="mt-3 text-2xl font-semibold text-zinc-950 dark:text-white">0.9s median</p>
              <p className="mt-2 text-xs leading-relaxed text-zinc-500 dark:text-zinc-450">
                Disk-backed embeddings and SQLite response cache.
              </p>
            </article>

            <article className="rounded-2xl border border-zinc-200/70 dark:border-zinc-800/80 bg-white/90 dark:bg-zinc-900/90 p-5 shadow-[0_16px_36px_-26px_rgba(15,23,42,0.25)] dark:shadow-[0_16px_36px_-26px_rgba(0,0,0,0.4)]">
              <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">Endpoints</p>
              <p className="mt-3 text-2xl font-semibold text-zinc-950 dark:text-white">5 routes</p>
              <p className="mt-2 text-xs leading-relaxed text-zinc-500 dark:text-zinc-450">
                /health, /simulate-review, /recommend, /users, /products
              </p>
            </article>
          </div>
        </div>
      </Section>

      <Section className="mx-auto max-w-[85rem] px-4 pb-24 sm:px-6 lg:px-12 relative z-10">
        <motion.div
          className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr]"
          variants={{ show: { transition: { staggerChildren: 0.12 } } }}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
        >
          {taskCards.map((card) => {
            const Icon = card.icon;
            return (
              <motion.article
                key={card.title}
                id={card.id}
                className="rounded-2xl border border-zinc-200/70 dark:border-zinc-800/80 bg-white/90 dark:bg-zinc-900/90 p-7 shadow-[0_24px_40px_-30px_rgba(15,23,42,0.3)] dark:shadow-[0_24px_40px_-30px_rgba(0,0,0,0.5)]"
                variants={{ hidden: { opacity: 0, y: 14 }, show: { opacity: 1, y: 0 } }}
                layout
              >
                <div className={`mb-4 inline-flex items-center gap-2 rounded-full border ${card.styles.badgeBorder} ${card.styles.badgeBg} px-3 py-1 text-xs font-medium ${card.styles.badgeText}`}>
                  <Icon size={14} weight="bold" />
                  {card.eyebrow}
                </div>
                <h3 className="text-2xl font-semibold tracking-tight text-zinc-950 dark:text-white">{card.title}</h3>
                <p className="mt-3 text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">{card.description}</p>
                <ul className="mt-6 space-y-2.5 text-sm text-zinc-600 dark:text-zinc-400">
                  {card.bullets.map((bullet) => (
                    <li key={bullet} className="flex items-start gap-2">
                      <span className={`mt-1.5 h-1.5 w-1.5 flex-shrink-0 rounded-full ${card.styles.bulletBg}`} />
                      <span>{bullet}</span>
                    </li>
                  ))}
                </ul>
                <div className="mt-8">
                  <AppLink
                    href={card.href}
                    className={`inline-flex items-center gap-2 text-sm font-semibold text-zinc-950 dark:text-white transition-colors ${card.styles.linkHover} focus:outline-none focus:ring-2 ${card.styles.ringFocus} focus:ring-offset-2`}
                  >
                    Open page <ArrowRight size={14} weight="bold" />
                  </AppLink>
                </div>
              </motion.article>
            );
          })}
        </motion.div>
      </Section>

      <Section id="api" className="mx-auto max-w-[85rem] px-4 pb-24 sm:px-6 lg:px-12 relative z-10">
        <div className="rounded-2xl border border-zinc-200/70 dark:border-zinc-800/80 bg-white/90 dark:bg-zinc-900/90 p-6 shadow-[0_20px_40px_-30px_rgba(15,23,42,0.3)] dark:shadow-[0_20px_40px_-30px_rgba(0,0,0,0.5)]">
          <div className="flex flex-col items-start justify-between gap-6 md:flex-row md:items-center">
            <div className="flex w-full items-start gap-3 md:w-auto md:items-center">
              <div className="shrink-0 rounded-xl border border-emerald-200/70 dark:border-emerald-900/50 bg-emerald-50 dark:bg-emerald-950/40 p-2.5 text-emerald-600 dark:text-emerald-400">
                <FileCode size={22} weight="bold" />
              </div>
              <div>
                <h4 className="text-sm font-semibold tracking-tight text-zinc-950 dark:text-white">API reference</h4>
                <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-450">
                  FastAPI exposes /health, /simulate-review, /recommend, /users, and /products.
                </p>
              </div>
            </div>

            <div className="flex w-full flex-wrap gap-2.5 md:w-auto">
              {routes.map((route) => (
                <span
                  key={route}
                  className="inline-flex items-center justify-center rounded-full border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-4 py-2 text-xs font-mono font-medium text-zinc-655 dark:text-zinc-350"
                >
                  {route}
                </span>
              ))}
            </div>
          </div>
        </div>
      </Section>

      <Section id="dataset" className="mx-auto max-w-[85rem] px-4 pb-24 sm:px-6 lg:px-12 relative z-10">
        <div className="grid gap-10 lg:grid-cols-[1.1fr_0.9fr]">
          <div>
            <div className="mb-4 inline-flex items-center gap-2 text-xs font-medium text-emerald-600 dark:text-emerald-450">
              <Database size={18} weight="bold" />
              <span className="uppercase tracking-widest">Data pipeline</span>
            </div>
            <h2 className="text-3xl font-semibold tracking-tight text-zinc-950 dark:text-white">
              Dataset strategy
            </h2>
            <p className="mt-3 max-w-2xl text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
              The repo builds profiles and indices from real review data, then warms the
              shared stores used by both LangGraph tasks.
            </p>
          </div>

          <div className="space-y-8 border-l border-zinc-200/70 dark:border-zinc-800/70 pl-6">
            {timeline.map((item) => (
              <div key={item.title} className="relative flex gap-4">
                <div className="mt-1.5 flex-shrink-0">
                  <div className="flex h-8 w-8 items-center justify-center rounded-full border border-emerald-300/70 dark:border-emerald-900/50 bg-emerald-50 dark:bg-emerald-950/40">
                    <div className="h-2 w-2 rounded-full bg-emerald-500" />
                  </div>
                </div>
                <div className="flex-1">
                  <span className="text-xs font-semibold uppercase tracking-widest text-emerald-600/80 dark:text-emerald-450/80">
                    {item.label}
                  </span>
                  <h3 className="mt-1 text-base font-semibold tracking-tight text-zinc-950 dark:text-white">
                    {item.title}
                  </h3>
                  <p className="mt-2 text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">{item.body}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </Section>

      <Section id="architecture" className="mx-auto max-w-[85rem] px-4 pb-24 sm:px-6 lg:px-12 relative z-10">
        <div className="rounded-2xl border border-zinc-200/70 dark:border-zinc-800/80 bg-white/90 dark:bg-zinc-900/90 p-8 shadow-[0_24px_40px_-30px_rgba(15,23,42,0.3)] dark:shadow-[0_24px_40px_-30px_rgba(0,0,0,0.5)]">
          <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">Architecture</p>
          <div className="mt-8 grid gap-4 sm:grid-cols-2">
            {architecture.map((item) => (
              <div key={item.label} className="rounded-xl border border-zinc-200/70 dark:border-zinc-800/70 bg-zinc-50/60 dark:bg-zinc-950/60 p-5">
                <div className="flex items-start gap-3">
                  <div className="rounded-xl border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-2 text-zinc-700 dark:text-zinc-300">
                    <item.icon size={20} weight="bold" />
                  </div>
                  <div>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">{item.label}</p>
                    <p className="text-sm font-semibold text-zinc-950 dark:text-white">{item.value}</p>
                    <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-450">{item.detail}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-8 rounded-xl border border-zinc-200/70 dark:border-zinc-800/70 bg-zinc-50/60 dark:bg-zinc-950/60 p-4 text-xs text-zinc-650 dark:text-zinc-400">
            <p>
              Default LLM: <span className="font-mono text-zinc-800 dark:text-zinc-200 font-semibold">qwen-plus</span>
            </p>
            <p>
              Embeddings: <span className="font-mono text-zinc-800 dark:text-zinc-200 font-semibold">all-MiniLM-L6-v2</span>
            </p>
            <p>
              Caches: <span className="font-mono text-zinc-800 dark:text-zinc-200 font-semibold">SQLite + disk-backed embedding cache</span>
            </p>
          </div>
        </div>
      </Section>

      <Section id="modules" className="mx-auto max-w-[85rem] px-4 pb-28 sm:px-6 lg:px-12 relative z-10">
        <div className="border-t border-zinc-200/70 dark:border-zinc-800/70 pt-10">
          <p className="mb-6 text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">
            Key modules
          </p>
          <div className="divide-y divide-zinc-200/70 dark:divide-zinc-850/70 rounded-2xl border border-zinc-200/70 dark:border-zinc-800/80 bg-white/90 dark:bg-zinc-900/90">
            {modules.map((module) => (
              <div key={module.name} className="flex flex-col gap-2 p-5 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="text-sm font-semibold text-zinc-950 dark:text-white">{module.name}</p>
                  <p className="mt-1 text-xs leading-relaxed text-zinc-500 dark:text-zinc-400">{module.role}</p>
                </div>
                <span className="inline-flex items-center gap-2 text-xs font-medium text-emerald-700 dark:text-emerald-400">
                  <PulseDot className="h-2 w-2" />
                  Active
                </span>
              </div>
            ))}
          </div>
        </div>
      </Section>
    </Shell>
  );
}

function TaskAPage() {
  const { users, loading } = useUsers();
  const [selectedUserId, setSelectedUserId] = useState("");
  const [guestUser, setGuestUser] = useState(null);
  const [reviewState, setReviewState] = useState({ status: "idle", data: null, error: null });
  const effectiveSelectedUserId = guestUser ? guestUser.user_id : (selectedUserId || users[0]?.user_id || "");
  const selectedUser = guestUser ? null : (users.find((user) => user.user_id === effectiveSelectedUserId) || null);
  const effectiveSelectedUser = guestUser
    ? { name: guestUser.name, review_count: 0, mean_rating: 0, top_category: "unknown", isGuest: true, persona: guestUser.persona }
    : selectedUser;

  return (
    <Shell
      footerText="Task A simulates how a user would rate and review a product using historical reviews, style analysis, and Naija voice injection."
      footerLinks={[
        { label: "Home page", href: "/" },
        { label: "Task B page", href: "/task-b" },
        { label: "Task A paper", href: PAPER_LINKS.taskA, external: true },
        { label: "Task B paper", href: PAPER_LINKS.taskB, external: true },
      ]}
      footerMeta="Output: SimulateReviewResponse with rating and review"
    >
      <Section id="top" className="relative mx-auto max-w-[85rem] px-4 pb-24 pt-32 sm:px-6 lg:px-12 relative z-10">
        <div className="inline-flex items-center gap-2 rounded-full border border-emerald-200/70 dark:border-emerald-900/50 bg-emerald-50/80 dark:bg-emerald-950/40 px-3 py-1 text-[10px] uppercase tracking-[0.22em] text-emerald-700 dark:text-emerald-450">
          <PulseDot />
          POST /simulate-review
        </div>

        <div className="mt-10 max-w-3xl">
          <h1 className="text-4xl font-semibold tracking-tighter leading-none text-zinc-950 dark:text-white md:text-6xl">
            <span className="text-emerald-600 dark:text-emerald-450 italic font-display text-[1.08em]">Task A</span> dashboard for user modelling.
          </h1>
          <p className="mt-6 max-w-xl text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
            Select a user, inspect their profile summary, then run the review simulation endpoint
            with product details.
          </p>
        </div>

        <div className="mt-8 flex flex-wrap gap-3">
          <MagneticButton
            type="button"
            className="group inline-flex items-center gap-2 rounded-xl bg-zinc-950 dark:bg-zinc-50 px-6 py-3 text-sm font-semibold text-white dark:text-zinc-950 shadow-[0_10px_26px_-16px_rgba(9,9,11,0.6)] dark:shadow-[0_10px_26px_-16px_rgba(255,255,255,0.15)] transition-transform hover:-translate-y-0.5 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:ring-offset-2"
            onClick={() => navigate("/")}
          >
            Back to home
            <ArrowRight size={16} weight="bold" className="transition group-hover:translate-x-1" />
          </MagneticButton>
          <AppLink
            className="inline-flex items-center gap-2 rounded-xl border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-5 py-3 text-sm font-semibold text-zinc-700 dark:text-zinc-300 transition-all hover:-translate-y-0.5 hover:border-zinc-300/70 dark:hover:border-zinc-700 hover:text-zinc-950 dark:hover:text-white focus:outline-none focus:ring-2 focus:ring-orange-500/40 focus:ring-offset-2"
            href="/task-b"
          >
            See Task B
            <ArrowRight size={14} weight="bold" className="text-orange-600" />
          </AppLink>
        </div>
      </Section>

      <UserDashboard
        title="Select a user profile"
        users={users}
        loading={loading}
        selectedUserId={effectiveSelectedUserId}
        onChange={(id) => { setGuestUser(null); setSelectedUserId(id); }}
        selectedUser={selectedUser}
        color="emerald"
        onGuestChange={setGuestUser}
        guestUser={guestUser}
      />

      <Section className="mx-auto max-w-[85rem] px-4 pb-24 sm:px-6 lg:px-12 relative z-10">
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(360px,0.9fr)]">
          <article className="rounded-2xl border border-zinc-200/70 dark:border-zinc-800/80 bg-white/80 dark:bg-zinc-900/80 p-6 shadow-[0_24px_40px_-30px_rgba(15,23,42,0.3)] dark:shadow-[0_24px_40px_-30px_rgba(0,0,0,0.5)] backdrop-blur sm:p-8">
            <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">Task A endpoint</p>
            <h2 className="mt-3 text-3xl font-semibold tracking-tight text-zinc-950 dark:text-white">
              Run the review simulation
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
              This section posts to <code className="rounded bg-zinc-100 dark:bg-zinc-800 px-1 py-0.5 text-[0.85em] text-zinc-800 dark:text-zinc-200">/api/simulate-review</code> and
              uses the selected user from the dashboard above.
            </p>
            <ReviewForm setReviewState={setReviewState} selectedUserId={effectiveSelectedUserId} selectedUser={effectiveSelectedUser} guestUser={guestUser} />
          </article>

          <ResultsGrid activeTab="review" reviewState={reviewState} />
        </div>
      </Section>
    </Shell>
  );
}

function TaskBPage() {
  const { users, loading } = useUsers();
  const [selectedUserId, setSelectedUserId] = useState("");
  const [guestUser, setGuestUser] = useState(null);
  const [recommendState, setRecommendState] = useState({ status: "idle", data: null, error: null });
  const effectiveSelectedUserId = guestUser ? guestUser.user_id : (selectedUserId || users[0]?.user_id || "");
  const selectedUser = guestUser ? null : (users.find((user) => user.user_id === effectiveSelectedUserId) || null);
  const effectiveSelectedUser = guestUser
    ? { name: guestUser.name, review_count: 0, mean_rating: 0, top_category: "unknown", isGuest: true, persona: guestUser.persona }
    : selectedUser;

  return (
    <Shell
      footerText="Task B adapts recommendations to real-time context using profile loading, hybrid retrieval, reranking, and multiturn refinement."
      footerLinks={[
        { label: "Home page", href: "/" },
        { label: "Task A page", href: "/task-a" },
        { label: "Task A paper", href: PAPER_LINKS.taskA, external: true },
        { label: "Task B paper", href: PAPER_LINKS.taskB, external: true },
      ]}
      footerMeta="Output: RecommendResponse with ranked recommendations"
    >
      <Section id="top" className="relative mx-auto max-w-[85rem] px-4 pb-24 pt-32 sm:px-6 lg:px-12 relative z-10">
        <div className="inline-flex items-center gap-2 rounded-full border border-orange-200/70 dark:border-orange-900/50 bg-orange-50/80 dark:bg-orange-950/40 px-3 py-1 text-[10px] uppercase tracking-[0.22em] text-orange-700 dark:text-orange-450">
          <PulseDot color="orange" />
          POST /recommend
        </div>

        <div className="mt-10 max-w-3xl">
          <h1 className="text-4xl font-semibold tracking-tighter leading-none text-zinc-950 dark:text-white md:text-6xl">
            <span className="text-orange-600 dark:text-orange-450 italic font-display text-[1.08em]">Task B</span> dashboard for contextual recommendations.
          </h1>
          <p className="mt-6 max-w-xl text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
            Select a user, inspect the profile dashboard, then run the recommendation
            endpoint with the current request and session history.
          </p>
        </div>

        <div className="mt-8 flex flex-wrap gap-3">
          <MagneticButton
            type="button"
            className="group inline-flex items-center gap-2 rounded-xl bg-zinc-950 dark:bg-zinc-50 px-6 py-3 text-sm font-semibold text-white dark:text-zinc-950 shadow-[0_10px_26px_-16px_rgba(9,9,11,0.6)] dark:shadow-[0_10px_26px_-16px_rgba(255,255,255,0.15)] transition-transform hover:-translate-y-0.5 focus:outline-none focus:ring-2 focus:ring-orange-500/50 focus:ring-offset-2"
            onClick={() => navigate("/")}
          >
            Back to home
            <ArrowRight size={16} weight="bold" className="transition group-hover:translate-x-1" />
          </MagneticButton>
          <AppLink
            className="inline-flex items-center gap-2 rounded-xl border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-5 py-3 text-sm font-semibold text-zinc-700 dark:text-zinc-300 transition-all hover:-translate-y-0.5 hover:border-zinc-300/70 dark:hover:border-zinc-700 hover:text-zinc-950 dark:hover:text-white focus:outline-none focus:ring-2 focus:ring-orange-500/40 focus:ring-offset-2"
            href="/task-a"
          >
            See Task A
            <ArrowRight size={14} weight="bold" className="text-orange-600" />
          </AppLink>
        </div>
      </Section>

      <UserDashboard
        title="Select a user profile"
        users={users}
        loading={loading}
        selectedUserId={effectiveSelectedUserId}
        onChange={(id) => { setGuestUser(null); setSelectedUserId(id); }}
        selectedUser={selectedUser}
        color="orange"
        onGuestChange={setGuestUser}
        guestUser={guestUser}
      />

      <Section className="mx-auto max-w-[85rem] px-4 pb-24 sm:px-6 lg:px-12 relative z-10">
        <div className="flex flex-col gap-6">
          <article className="rounded-2xl border border-zinc-200/70 dark:border-zinc-800/80 bg-white/80 dark:bg-zinc-900/80 p-6 shadow-[0_24px_40px_-30px_rgba(15,23,42,0.3)] dark:shadow-[0_24px_40px_-30px_rgba(0,0,0,0.5)] backdrop-blur sm:p-8">
            <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">Task B endpoint</p>
            <h2 className="mt-3 text-3xl font-semibold tracking-tight text-zinc-950 dark:text-white">
              Run the recommendation engine
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
              This section posts to <code className="rounded bg-zinc-100 dark:bg-zinc-800 px-1 py-0.5 text-[0.85em] text-zinc-800 dark:text-zinc-200">/api/recommend</code> and
              uses the selected user from the dashboard above.
            </p>
            <RecommendForm
              setRecommendState={setRecommendState}
              selectedUserId={effectiveSelectedUserId}
              selectedUser={effectiveSelectedUser}
              guestUser={guestUser}
              recommendState={recommendState}
            />
          </article>

          <ResultsGrid activeTab="recommend" recommendState={recommendState} />
        </div>
      </Section>
    </Shell>
  );
}

function NotFoundPage() {
  return (
    <Shell
      footerText="The page you requested does not exist. Use the navigation above to return to the task pages."
      footerLinks={[
        { label: "Home page", href: "/" },
        { label: "Task A page", href: "/task-a" },
        { label: "Task B page", href: "/task-b" },
        { label: "Task A paper", href: PAPER_LINKS.taskA, external: true },
      ]}
      footerMeta="404 - route not found"
    >
      <Section className="relative mx-auto max-w-[85rem] px-4 pb-32 pt-32 sm:px-6 lg:px-12 relative z-10">
        <div className="max-w-2xl">
          <p className="text-[10px] uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">404</p>
          <h1 className="mt-4 text-4xl font-semibold tracking-tighter text-zinc-950 dark:text-white md:text-6xl">
            That route does not exist.
          </h1>
          <p className="mt-6 max-w-xl text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
            Ego only exposes the home page, Task A, and Task B. Use the navigation to
            return to a valid route.
          </p>
          <div className="mt-10">
            <MagneticButton
              type="button"
              className="inline-flex items-center gap-2 rounded-xl bg-zinc-950 dark:bg-zinc-50 px-6 py-3 text-sm font-semibold text-white dark:text-zinc-950 shadow-[0_10px_26px_-16px_rgba(9,9,11,0.6)] dark:shadow-[0_10px_26px_-16px_rgba(255,255,255,0.15)] transition-transform hover:-translate-y-0.5 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:ring-offset-2"
              onClick={() => navigate("/")}
            >
              Return home
              <ArrowRight size={16} weight="bold" />
            </MagneticButton>
          </div>
        </div>
      </Section>
    </Shell>
  );
}

export default function App() {
  const pathname = usePathname();

  if (pathname === "/") return <HomePage />;
  if (pathname === "/task-a") return <TaskAPage />;
  if (pathname === "/task-b") return <TaskBPage />;
  return <NotFoundPage />;
}
