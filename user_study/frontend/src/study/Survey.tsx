import { useState } from "react";

const ratings = [
  ["manual_confidence", "Manual: confidence in detecting falls"], ["manual_effort", "Manual: mental effort required"], ["manual_ease", "Manual: ease of monitoring multiple feeds"],
  ["automated_confidence", "Automated: confidence in detecting falls"], ["automated_effort", "Automated: mental effort required"], ["automated_ease", "Automated: ease of monitoring"],
  ["trust", "Trust in automated alerts"],
];
export default function Survey({ submit, busy }: { submit: (body: unknown) => Promise<void>; busy: boolean }) {
  const [values, setValues] = useState<Record<string, number>>({});
  const [feedback, setFeedback] = useState("");
  return <section className="study-panel"><h2>Final survey</h2><p>Allow about four minutes. Rate each item from 1, very low, to 5, very high. Higher effort means more work.</p>
    <form onSubmit={e => { e.preventDefault(); void submit({ ...values, feedback }); }}>
      {ratings.map(([name, label]) => <fieldset key={name}><legend>{label}</legend><div className="study-ratings">{[1, 2, 3, 4, 5].map(n => <label key={n}><input type="radio" required name={name} value={n} checked={values[name] === n} onChange={() => setValues(v => ({ ...v, [name]: n }))} />{n}</label>)}</div></fieldset>)}
      <label>What was difficult or confusing? What would you improve?<textarea maxLength={3000} value={feedback} onChange={e => setFeedback(e.target.value)} /></label>
      <button className="btn study-primary" disabled={busy}>Submit survey and finish</button>
    </form>
  </section>;
}
