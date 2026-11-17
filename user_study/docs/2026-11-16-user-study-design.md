# User study design

Date: 2026-11-16
Branch: user-study

## Purpose

Run the study in the supplied user-study procedures PDF inside the existing React dashboard. Compare manual monitoring with automated alerts for the same participant, using different, balanced prerecorded video sets. Keep the normal dashboard available outside study sessions.

## Proposed approach

Add a dedicated User study screen backed by isolated study records and endpoints in FastAPI. Use prerecorded raw footage with separately generated detector alerts. Generate alerts with the existing fall pipeline before a session and replay them at their measured video timestamps. Never derive alerts from ground-truth labels. This allows repeatable simultaneous camera playback without requiring a laptop to run inference on several feeds in real time.

Alternatives are running inference live on every study feed, which adds hardware-dependent delays, or building a separate study application, which duplicates the dashboard and review controls. The recommendation is integrated prerecorded replay. This approach has not yet been approved.

## Session flow

1. Facilitator configures two distinct video sets, raw media, camera identifiers, ground-truth fall timestamps, detector alert timestamps, detector version, and a matching tolerance. Validate configuration before allowing a session. Ground truth must never appear on participant screens.
2. Facilitator enters an anonymous participant code, Oxy worker or student group, age range, monitoring experience, eligibility, and consent confirmation. Names and signed consent forms stay outside study results. Explain voluntary participation and withdrawal. Consent text is a study draft, not an assertion of institutional approval.
3. Assign one of four sequences across participants: manual A then automated B; automated B then manual A; manual B then automated A; automated A then manual B. Record the assignment and support balancing within each participant group.
4. Allow about six minutes for consent, instructions, and practice. Practice footage is separate from test footage.
5. Each condition lasts ten minutes. Start camera videos together after all feeds are ready. Hide seek and speed controls. Stop the study clock when playback stalls or the page is hidden and record interruptions. Aborted conditions are incomplete, never scored as completed.
6. Manual condition displays raw simultaneous feeds without overlays or detector alerts. A participant reports a possible fall, selects a camera, and records an approximate incident timestamp. Also record when the report was submitted so recalled incident time cannot be mistaken for detection latency.
7. Automated condition provides an unrelated task while scheduled detector alerts appear. Participants acknowledge, review incident footage, and choose an intended action such as dispatch help, investigate, or dismiss as a false alarm. Record alert emission, acknowledgment, and decision separately. No action contacts external responders.
8. Finish with a roughly four-minute survey: confidence, mental effort, ease of monitoring for each condition, trust in automation, and open feedback. Facilitator observations are stored separately from participant answers.
9. Show a facilitator-only results view and export anonymous raw records and computed metrics. Withdrawal removes participant records and excludes the participant from aggregate results.

## Measurement rules

Match reports or detector alerts to ground truth by camera and configured time window, one detection per fall and one fall per detection. Retain duplicates in raw logs and count unmatched detections as false alarms. Report detected and missed falls, false alarms, precision, recall, and latency in seconds for each participant and condition. Show undefined ratios and absent latency samples as unavailable, never as zero. Distinguish system detection latency from participant acknowledgment and decision latency. Compare Oxy workers and students separately and combined. Preserve sample sizes, incomplete sessions, counterbalancing assignments, model version, and interruptions in exports.

## Storage and privacy

Use separate study storage rather than production Event rows. Restrict facilitator configuration and results using a configured study facilitator secret, with participant sessions receiving narrowly scoped tokens. Keep participant identity outside this application. Do not expose label files, other participants, or facilitator notes through participant endpoints. Deployment still requires the team's protected hosting and restricted access described in the PDF; a local development server alone does not provide that protection.

## Implementation boundaries

- user_study/frontend/src/study/: validated configuration, participant session flow, synchronized players, alert review, survey, results, and styles.
- ui/src/App.tsx: navigation between Dashboard and User study.
- user_study/backend/: configuration, state transitions, scoped authorization, storage, scoring, export, and media access.
- src/backend/main.py: register study endpoints.
- user_study/scripts/: adapter to produce study alert schedules from actual pipeline output.
- tests/: counterbalancing, one-to-one scoring, participant authorization, timestamp validation, session transitions, exports, and withdrawal.

Do not change detector training, apply the saved training stash, upload results externally, recruit participants, or claim study findings before real sessions exist.

## Validation

Run focused backend tests and the frontend production build. Exercise a short test configuration in the browser through both conditions, report submission, alert acknowledgment and review, survey, export, playback failure, and withdrawal. Verify manual mode contains no detector cues and participant endpoints reveal no ground truth. Use clearly labeled test fixtures only; actual study footage and model outputs must be supplied and verified by the facilitator before recruitment.

## Review refinements

This is a replay-based workflow evaluation. Video timestamps measure algorithmic recognition delay within footage; they do not establish live inference or alert delivery latency. Export that distinction explicitly. Any claim about live end-to-end response speed requires separately measured processing and delivery delays or a live study configuration.

Use deterministic maximum-cardinality bipartite matching within explicit early and late tolerance limits, resolving ties by smallest total absolute time difference and then stable event identifiers. Record both monotonic study time and wall-clock interruption intervals. Use the same unrelated task for every automated condition: a repeating arithmetic worksheet with generated operands, recording answers, correctness, and answer timestamps without changing alert visibility.
