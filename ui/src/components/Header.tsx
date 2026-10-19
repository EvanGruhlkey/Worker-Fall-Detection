import { useEffect, useState } from "react";
import { LOCATION } from "../data/mock";
import Icon from "./Icon";

function useClock(): string {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now.toLocaleString("en-US", {
    month: "2-digit", day: "2-digit", year: "numeric",
    hour: "numeric", minute: "2-digit", second: "2-digit",
  });
}

/** Fake macOS title bar. Remove <TitleBar /> from App if the app is wrapped in a native window. */
export function TitleBar() {
  return (
    <div className="titlebar">
      <div className="lights">
        <span className="l r" /><span className="l y" /><span className="l g" />
      </div>
      <div className="title"><Icon name="fall" size={14} />Oxy · Fall detection</div>
    </div>
  );
}

export function TopBar() {
  const clock = useClock();
  return (
    <header className="topbar">
      <div className="loc">
        <Icon name="pin" />
        <span>{LOCATION.name}</span>
        <span className="coords">{LOCATION.coords}</span>
      </div>
      <div className="right">
        <div className="clock"><Icon name="clock" />{clock}</div>
        <button className="settings"><Icon name="gear" size={16} />Settings</button>
      </div>
    </header>
  );
}
