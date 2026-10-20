import { useDashboard, useReportLocation } from "./api";
import CameraFeed from "./components/CameraFeed";
import EventLog from "./components/EventLog";
import { TitleBar, TopBar } from "./components/Header";
import StatCard from "./components/StatCard";

export default function App() {
  useReportLocation();
  const { location, stats, camera, events } = useDashboard();

  return (
    <div className="window">
      <TitleBar />
      <TopBar location={location} />
      <main className="content">
        <h1>Dashboard</h1>
        <p className="sub">Real-time worker safety monitoring</p>

        <section className="stats">
          {stats.map((s) => <StatCard key={s.label} {...s} />)}
        </section>

        <section className="grid">
          <div className="feed-col">
            <h2>Camera feed <span className="count">1</span></h2>
            <div style={{ marginTop: 12 }}>
              <CameraFeed camera={camera} location={location} />
            </div>
          </div>
          <EventLog events={events} />
        </section>
      </main>
    </div>
  );
}
