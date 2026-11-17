import { createRoot } from "react-dom/client";
import Study from "./study/Study";
import "./base.css";

const root = document.getElementById("root");
if (root) createRoot(root).render(<Study />);
