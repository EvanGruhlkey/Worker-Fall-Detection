import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import GamePage from "./pages/GamePage";
import { useGameStore } from "./store/useGameStore";
import "./index.css";

window.addEventListener("message", event => {
  if (event.source !== window.parent || event.origin !== window.location.origin ||
      event.data?.type !== "study:runner" || typeof event.data.enabled !== "boolean") return;
  useGameStore.setState({ studyEnabled: event.data.enabled });
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <GamePage />
  </StrictMode>
);
