import { useState } from "react";

type Condition = "manual" | "automated";

const questions = [
  ["confidence", "How confident were you that you caught falls?"],
  ["effort", "How much mental effort did this part take?"],
  ["ease", "How easy was it to monitor for falls?"],
] as const;

export default function Survey({ lastCondition, submit, busy }: {
  lastCondition: Condition;
  submit: (body: unknown) => Promise<void>;
  busy: boolean;
}) {
  const [values, setValues] = useState<Record<string, number>>({});
  const [feedback, setFeedback] = useState("");
  const order: Condition[] = lastCondition === "manual" ? ["automated", "manual"] : ["manual", "automated"];

  function rating(name: string, question: string) {
    return <fieldset key={name}>
      <legend>{question}</legend>
      <div className="study-ratings">{[1, 2, 3, 4, 5].map(value => <label key={value}>
        <input type="radio" required name={name} value={value} checked={values[name] === value}
          onChange={() => setValues(previous => ({ ...previous, [name]: value }))} />
        {value}
      </label>)}</div>
    </fieldset>;
  }

  return <section className="study-panel">
    <h2>Final survey</h2>
    <p>For each question, choose 1 for “not at all” or 5 for “very much.”</p>
    <form onSubmit={event => { event.preventDefault(); void submit({ ...values, feedback }); }}>
      {order.map((condition, index) => <section key={condition} aria-labelledby={`survey-${condition}`}>
        <h3 id={`survey-${condition}`}>Part {index + 1} · {condition === "manual" ? "Manual camera monitoring" : "Automated fall alerts"}</h3>
        {questions.map(([name, question]) => rating(`${condition}_${name}`, question))}
        {condition === "automated" && rating("trust", "How much did you trust the fall alerts?")}
      </section>)}
      <label>Anything you’d change? (optional)
        <textarea rows={2} maxLength={3000} value={feedback} onChange={event => setFeedback(event.target.value)} />
      </label>
      <button className="btn study-primary" disabled={busy}>Finish study</button>
    </form>
  </section>;
}
