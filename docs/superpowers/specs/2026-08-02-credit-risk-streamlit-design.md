# Credit Risk Streamlit Demonstration Design

## Purpose

Build a read-only portfolio demonstration for hiring managers, risk modelers, and data
scientists. The first screen must be the usable credit-risk workflow, not a marketing page.
It must make the calibrated prediction, policy action, evidence, and limitations easy to
inspect without implying that the tool is suitable for real lending decisions.

## Selected Direction

Use a quiet operational work surface with application inputs on the left and the decision
result on the right at desktop widths. On narrow screens, use one column with inputs first,
the command next, and the decision result immediately afterward. Evidence tabs follow the
primary workflow at every width.

Alternatives considered:

- A card-heavy dashboard was rejected because it fragments a repeated analytical workflow.
- A guided multi-step form was rejected because the synthetic demonstration should be fast
  to scan and compare.
- A marketing-style hero with a product preview was rejected because the application itself
  is the portfolio evidence.

## Visual References

- Desktop concept:
  `/Users/joyboy/.codex/visualizations/2026/08/01/019fbdfa-0a30-7810-9de9-959646569c93/credit-risk-streamlit-concepts/desktop-concept.png`
- Mobile concept:
  `/Users/joyboy/.codex/visualizations/2026/08/01/019fbdfa-0a30-7810-9de9-959646569c93/credit-risk-streamlit-concepts/mobile-concept.png`

The concepts define hierarchy, density, spacing, and component treatment. Their sample
probabilities, policy thresholds, model version, dates, and performance metrics are not
product facts. The implementation must read all values from the validated release bundle
or show a clear unavailable state. Do not implement the concept's settings action, invented
model version, or invented data date.

## Information Architecture

1. Header: `Credit Risk Decision Lab` and a quiet `Release bundle verified` state.
2. Persistent warning: `Demonstration only - not a lending decision system. Inputs are not stored.`
3. Application inputs: all `CreditApplication` fields, pre-populated with a synthetic example.
4. Primary command: `Run assessment`.
5. Decision result: calibrated default probability, policy action, threshold context, and
   local explanation labeled `Associations, not causal effects`.
6. Evidence tabs in this order: `Model Performance`, `Calibration`, `Business Cost`,
   `Fairness`, `Limitations`.
7. Startup failure: a blocking, actionable error when the bundle is absent, inconsistent,
   unsafe, or cannot be loaded.

## Interaction And Data Flow

- Load and validate `artifacts/release/release_manifest.json` and every referenced artifact
  through the release-bundle API before rendering the prediction workflow.
- Cache immutable artifact loading, not user-entered values or prediction results.
- Build a one-row frame from a validated `CreditApplication`, transform it with the frozen
  preprocessor, and score it with the frozen calibrated model.
- Select the bad-class probability using `model.classes_`; never assume probability column 1.
- Assign the action from the frozen policy using the established policy implementation.
- Return a strict `CreditPrediction`.
- Show local SHAP associations from the release explanation payload without presenting them
  as causal effects. If the payload cannot support the entered case, show the documented
  global/example explanation rather than fabricating local values.
- Do not write input values to disk, session logs, analytics, or artifacts.

## Visual System

- Background: true white `#FFFFFF`; secondary band `#F6F8FA`.
- Text: near-black `#17202A`; muted text `#5F6B76`; borders `#D7DEE4`.
- Accent: restrained teal `#078A8C`; semantic approve `#2E9B55`, review `#D99A00`, decline
  `#D64545`.
- Typography: compact modern sans serif, no hero-scale type, no viewport-scaled font sizes,
  letter spacing `0`.
- Geometry: maximum 6px radius, thin borders, minimal shadow, stable field and button heights.
- Container model: open bands, dividers, rows, tables, and chart frames. No nested cards,
  floating section cards, bento grid, gradients, decorative pills, or ornamental imagery.

## Responsive Behavior

- Desktop: two stable columns, approximately 48/52, with input and result regions aligned.
- Tablet: two columns may remain when controls fit; otherwise collapse without horizontal
  page overflow.
- Mobile: one column, touch-friendly controls, result directly below the command, long feature
  names wrapping cleanly, and a horizontally scrollable tab strip.
- Tables may become compact lists on mobile when that prevents clipped columns.
- Charts must use responsive widths and preserve readable axes.

## Error Handling

- Missing or inconsistent release artifacts stop the app before scoring and identify the
  invalid bundle without exposing a traceback to the user.
- Invalid form input is shown beside the workflow and never sent to model code.
- Prediction errors fail closed with a demonstration error; no partial probability or action
  is displayed.
- Missing optional display data produces an explicit unavailable message, never a made-up
  metric.

## Verification

- Import and application tests cover artifact loading, strict schema use, class-aware
  probability selection, policy assignment, no persistence, and failure states.
- Run the app against a deterministic test release bundle.
- Verify desktop at 1536x1024 and mobile near 390x844 in a browser.
- Capture screenshots and compare them with both concept images using `view_image`.
- Check copy, hierarchy, palette, typography, spacing, container model, responsive behavior,
  and the assessment interaction before accepting the implementation.

