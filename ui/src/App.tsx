import { useState } from "react";
import { useDashboard, useReportLocation } from "./api";
import CameraFeed from "./components/CameraFeed";
import EventLog from "./components/EventLog";
import EventReview from "./components/EventReview";
import { TitleBar, TopBar } from "./components/Header";
import StatCard from "./components/StatCard";
import VideoUpload from "./components/VideoUpload";
import type { EventScope, FallEvent } from "./types";

export default function App() {
  useReportLocation();
  const [scope, setScope] = useState<EventScope>("today");
  const { data, refresh } = useDashboard(scope);
  const { location, stats, camera, events, uploads } = data;

  // Keep the latest copy of the selected event (status, clip URL) as data refreshes.
  const [selected, setSelected] = useState<FallEvent | null>(null);
  const reviewing = selected && (events.find((e) => e.id === selected.id) ?? selected);

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
            <VideoUpload uploads={uploads} onUploaded={refresh} />
          </div>
          <EventLog events={events} scope={scope} onScopeChange={setScope} onSelect={setSelected} />
        </section>
      </main>
      {reviewing && (
        <EventReview
          key={reviewing.id}
          event={reviewing}
          onClose={() => setSelected(null)}
          onChanged={refresh}
        />
      )}
    </div>
  );
}
