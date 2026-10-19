import type { IconName } from "../types";

const PATHS: Record<IconName, string> = {
  alert: "M12 4l9 16H3z M12 10v4 M12 17h.01",
  warn: "M12 4l9 16H3z M12 10v4 M12 17h.01",
  cam: "M3 8a2 2 0 012-2h9a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z M16 10l5-3v10l-5-3z",
  pulse: "M3 12h4l3-8 4 16 3-8h4",
  shield: "M12 3l7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6z M9 12l2 2 4-4",
  fall: "M12 3l7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6z M12 9v4 M12 16h.01",
  ok: "M3 12a9 9 0 1018 0 9 9 0 10-18 0 M8 12l3 3 5-6",
  eye: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z M9 12a3 3 0 106 0 3 3 0 10-6 0",
  pin: "M12 21s7-6 7-11a7 7 0 10-14 0c0 5 7 11 7 11z M9.5 10a2.5 2.5 0 105 0 2.5 2.5 0 10-5 0",
  clock: "M3 12a9 9 0 1018 0 9 9 0 10-18 0 M12 7v5l3 2",
  gear: "M9 12a3 3 0 106 0 3 3 0 10-6 0 M19 12a7 7 0 00-.1-1.2l2-1.5-2-3.4-2.3.9a7 7 0 00-2-1.2L14.2 3h-4l-.4 2.6a7 7 0 00-2 1.2l-2.3-.9-2 3.4 2 1.5a7 7 0 000 2.4l-2 1.5 2 3.4 2.3-.9a7 7 0 002 1.2l.4 2.6h4l.4-2.6a7 7 0 002-1.2l2.3.9 2-3.4-2-1.5c.1-.4.1-.8.1-1.2z",
  chevron: "M6 9l6 6 6-6",
  download: "M12 4v11 M7 11l5 5 5-5 M5 20h14",
  wifi: "M5 12a10 10 0 0114 0 M8.5 15.5a5 5 0 017 0 M12 19h.01",
};

interface Props {
  name: IconName;
  size?: number;
  className?: string;
}

export default function Icon({ name, size = 14, className }: Props) {
  return (
    <svg
      className={`ic${className ? ` ${className}` : ""}`}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
