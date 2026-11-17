# User study implementation plan

Approved design: docs/superpowers/specs/2026-11-16-user-study-design.md.

1. Build isolated SQLite study storage, validated study manifests, scoped authorization, counterbalanced assignments, session transitions, event logging, deterministic one-to-one scoring, and exports. Verify through standalone API tests without production database or model startup.
2. Add a dedicated React study screen with facilitator configuration and session creation, participant consent and instructions, synchronized raw feeds, timed conditions, manual reporting, automated review, a standardized arithmetic task, survey, interruption handling, withdrawal, and exports. Run the production build.
3. Add a pipeline-alert schedule adapter, a clearly marked short demo manifest and media generator, and operating instructions. Exercise the running frontend and API with the demo.
4. Review the complete change, address material findings, rerun affected checks, and report remaining operational requirements.

Ruling: use a separate SQLite file under FALL_STUDY_DIR rather than production PostgreSQL. This isolates personal study records and allows study endpoints to be tested without model or production DB startup. Restricted hosting and protected filesystem access remain operator responsibilities.
Ruling: store one immutable manifest snapshot per session and exclude it from participant responses. Facilitator routes require FALL_STUDY_SECRET; no default secret is enabled.

## Execution record

- Backend, UI, demo generator, adapter, and operating guide implemented.
- Reviewer found media-stream transaction scope, interrupted-client recovery, dropped safety pauses, and cross-condition caching risks. Fixed these at the dependency, state-machine, request-queue, and media URL/header boundaries.
- Browser testing found seeking during resume could trigger an unconditional buffering pause; now only running playback triggers that handler.
- Focused tests cover real API flows without production database startup. Browser demo exercised both condition transitions, manual reporting, pause/resume, operator task, survey, and session completion.

- User requested human-recorded footage and a plainer interface. Replaced the synthetic demo generator with recorded GMDCSA24 preparation, actual model inference, and a pilot mode excluded from study summaries. Reduced visual cards, repeated explanations, and raw JSON exposure; moved configuration behind an explicit disclosure.

- User clarified all cameras must be visible simultaneously. Both manual and automated conditions now use one persistent, two-column camera wall sized to the viewport. Browser inspection verified two real feeds playing at the same timestamp and both fully visible. Acknowledgment and intended response were also verified against a real classifier alert.

- User replaced the recorded-footage pilot with a 3x3 AI-video wall and clarified the goal is a ten-minute condition, rather than a response quota. Added nine-camera manifests, cycling rig-video preparation, and twenty staggered fall occurrences per condition. Actual per-clip classifier outputs are replayed with explicit reset/provenance metadata.

- Nine simultaneous media requests exposed SQLite connection thread affinity across FastAPI dependency entry/finalization. Transactions still own separate connections, with thread affinity disabled so request cleanup can complete on another worker. Added parallel ranged-media and polling regression coverage.

- User supplied a visual reference. Replaced the plain monitor layout with a full-screen safety console: derrick header, top-right timer, 3x3 camera wall with CAM badges, instructions, red reporting action, and event table. Camera tiles select the report source. Optional notes are edited after reporting so they cannot delay the original timestamp.

- Browser verification showed seven raw players advancing while two stalled on play despite all sources loading. Added a common 3x3 transport video, drawn into nine individual camera canvases. This preserves the nine source feeds and exact shared time while using one browser decoder. Original camera reels remain the review sources.

- User reduced the layout to two rows of three and asked to simplify the UI and code. Rebuilt the timeline for six cameras, replaced per-camera canvas drawing with one visible video and six overlays, removed decorative SVGs, duplicate media handlers, obsolete CSS, and mock practice screens. Previous nine-camera data remains in its separate ignored directory.

- User shortened each condition to five minutes. Rebuilt both six-camera sets to 300 seconds with all twenty labeled falls redistributed inside each set; regenerated alerts and synchronized video from the new offsets. Kept prior ten-minute records and media in their original directory.

- User shortened each condition again to two minutes. Rebuilt both six-camera sets to 120 seconds and redistributed the twenty labeled fall occurrences and actual detector alerts. Previous five-minute assets and records remain in their own directory.

- User requested a Zetamac-style task as the main automated view, with audible popup alerts. Added mixed arithmetic and automatic correct-answer advancement, server-validated question indices and scoring, native modal incident review, queued alerts, and a short Web Audio chime enabled by the Start gesture. Manual monitoring remains unchanged.
