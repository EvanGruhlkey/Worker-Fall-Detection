import { useEffect } from "react";
import { TEXT_SCALES, type Settings, type Theme } from "../useSettings";
import Icon from "./Icon";

interface Props {
  settings: Settings;
  onChange: (patch: Partial<Settings>) => void;
  onReset: () => void;
  onClose: () => void;
}

const THEMES: { value: Theme; label: string }[] = [
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
];

export default function SettingsPanel({ settings, onChange, onReset, onClose }: Props) {
  const { theme, textStep } = settings;

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="backdrop" onClick={onClose}>
      <div
        className="review settings-panel"
        role="dialog"
        aria-modal="true"
        aria-label="Settings"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="review-head">
          <h2>Settings</h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close settings">
            <Icon name="close" size={16} />
          </button>
        </div>

        <div className="setting-row">
          <div>
            <strong>Appearance</strong>
            <small>Light or dark interface</small>
          </div>
          <div className="segmented" role="group" aria-label="Theme">
            {THEMES.map((t) => (
              <button
                key={t.value}
                className={theme === t.value ? "active" : ""}
                aria-pressed={theme === t.value}
                onClick={() => onChange({ theme: t.value })}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        <div className="setting-row">
          <div>
            <strong>Text size</strong>
            <small>Make text smaller or larger</small>
          </div>
          <div className="actions">
            <button
              className="btn"
              onClick={() => onChange({ textStep: textStep - 1 })}
              disabled={textStep === 0}
              aria-label="Decrease text size"
            >
              A−
            </button>
            <button
              className="btn"
              onClick={() => onChange({ textStep: textStep + 1 })}
              disabled={textStep === TEXT_SCALES.length - 1}
              aria-label="Increase text size"
            >
              A+
            </button>
          </div>
        </div>

        <div className="review-actions">
          <button className="btn" onClick={onReset}>Reset to defaults</button>
          <button className="btn" onClick={onClose}>Done</button>
        </div>
      </div>
    </div>
  );
}
