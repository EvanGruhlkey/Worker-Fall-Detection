import { useLayoutEffect, useRef, useState, type RefObject } from "react";

export type TourStep = { target: string; title: string; text: string };

export default function RoundTour({ step, index, total, part, root, onBack, onNext }: {
  step: TourStep;
  index: number;
  total: number;
  part: number;
  root: RefObject<HTMLElement>;
  onBack: () => void;
  onNext: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const card = useRef<HTMLDivElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const [position, setPosition] = useState<{ rect: DOMRect; top: number; left: number }>();

  useLayoutEffect(() => {
    const modal = dialog.current;
    modal?.showModal();
    return () => modal?.close();
  }, []);

  useLayoutEffect(() => {
    const target = root.current?.querySelector<HTMLElement>(`[data-tour="${step.target}"]`);
    const callout = card.current;
    if (!target || !callout) return;
    target.scrollIntoView({ block: "nearest" });
    heading.current?.focus({ preventScroll: true });

    function measure() {
      if (!target || !callout) return;
      const rect = target.getBoundingClientRect();
      const { offsetHeight: height, offsetWidth: width } = callout;
      const below = rect.bottom + 14;
      const above = rect.top - height - 14;
      const top = below + height <= window.innerHeight - 12 ? below
        : above >= 12 ? above : Math.max(12, window.innerHeight - height - 12);
      const left = Math.min(window.innerWidth - width - 12, Math.max(12, rect.left + (rect.width - width) / 2));
      setPosition({ rect, top, left });
    }

    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(target);
    observer.observe(callout);
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure, true);
    };
  }, [root, step]);

  const rect = position?.rect;
  return <dialog ref={dialog} className="round-tour" aria-labelledby="round-tour-title" onCancel={event => event.preventDefault()}>
    {rect && <div className="round-tour-highlight" aria-hidden="true" style={{
      top: rect.top - 4, left: rect.left - 4, width: rect.width + 8, height: rect.height + 8,
    }} />}
    <div ref={card} className="round-tour-card" style={{ top: position?.top ?? 12, left: position?.left ?? 12 }}>
      <h2 id="round-tour-title" ref={heading} tabIndex={-1}>{step.title}</h2>
      <p>{step.text}</p>
      <div className="round-tour-actions">
        <span className="round-tour-progress" aria-label={`Part ${part}, step ${index + 1} of ${total}`}>{index + 1} of {total}</span>
        {index > 0 && <button className="study-text-button" onClick={onBack}>Back</button>}
        <button className="btn" onClick={onNext}>{index + 1 === total ? "Done" : "Next"}</button>
      </div>
    </div>
  </dialog>;
}
