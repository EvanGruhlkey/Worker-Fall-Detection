import CameraFeed from "./components/CameraFeed";
import EventLog from "./components/EventLog";
import { TitleBar, TopBar } from "./components/Header";
import StatCard from "./components/StatCard";
import { CAMERA, EVENTS, STATS } from "./data/mock";

export default function App() {
  return (
    <div className="window">
      <TitleBar />
      <TopBar />
      <main className="content">
        <h1>Dashboard</h1>
        <p className="sub">Real-time worker safety monitoring</p>

        <section className="stats">
          {STATS.map((s) => <StatCard key={s.label} {...s} />)}
        </section>

        <section className="grid">
          <div className="feed-col">
            <h2>Camera feed <span className="count">1</span></h2>
            <div style={{ marginTop: 12 }}>
              <CameraFeed camera={CAMERA} />
            </div>
          </div>
          <EventLog events={EVENTS} />
        </section>
      </main>
    </div>
  );
}
