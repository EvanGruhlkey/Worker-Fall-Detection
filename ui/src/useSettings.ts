import { useCallback, useEffect, useState } from "react";

export type Theme = "dark" | "light";

export interface Settings {
  theme: Theme;
  /** Index into TEXT_SCALES. */
  textStep: number;
}

/** Multipliers applied to every font size in the app. */
export const TEXT_SCALES = [0.85, 0.92, 1, 1.12, 1.25, 1.4];
export const DEFAULT_TEXT_STEP = 2; // 100%

const DEFAULTS: Settings = { theme: "dark", textStep: DEFAULT_TEXT_STEP };
const STORAGE_KEY = "oxy-ui-settings";

function load(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      const stepOk =
        Number.isInteger(parsed.textStep) &&
        parsed.textStep >= 0 &&
        parsed.textStep < TEXT_SCALES.length;
      return {
        theme: parsed.theme === "light" ? "light" : "dark",
        textStep: stepOk ? parsed.textStep : DEFAULT_TEXT_STEP,
      };
    }
  } catch {
    // storage blocked or corrupt: fall back to defaults
  }
  return DEFAULTS;
}

/**
 * Display settings (theme and text size). Saved in the browser and applied to
 * <html> as `data-theme` and the `--text-scale` CSS variable.
 */
export function useSettings() {
  const [settings, setSettings] = useState<Settings>(load);

  useEffect(() => {
    const root = document.documentElement;
    root.dataset.theme = settings.theme;
    root.style.setProperty("--text-scale", String(TEXT_SCALES[settings.textStep]));
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      // ignore: settings just won't persist
    }
  }, [settings]);

  const update = useCallback(
    (patch: Partial<Settings>) => setSettings((prev) => ({ ...prev, ...patch })),
    [],
  );
  const reset = useCallback(() => setSettings(DEFAULTS), []);

  return { settings, update, reset };
}
