const API_BASE = window.location.hostname === "127.0.0.1" && window.location.port === "8000"
  ? ""
  : "/api";

const phqQuestions = [
  "Little interest or pleasure in doing things",
  "Feeling down, depressed, or hopeless",
  "Trouble falling or staying asleep, or sleeping too much",
  "Feeling tired or having little energy",
  "Poor appetite or overeating",
  "Feeling bad about yourself or that you are a failure or have let yourself or your family down",
  "Trouble concentrating on things, such as reading the newspaper or watching television",
  "Moving or speaking so slowly that other people could have noticed, or the opposite: feeling restless and unable to sit still",
  "Thoughts that you would be better off dead, or of hurting yourself in some way",
];

const phqOptions = [
  ["Not at all", 0],
  ["Several days", 1],
  ["More than half the days", 2],
  ["Nearly every day", 3],
];

// ...existing JS content...
// For brevity, the rest of app.js is identical to the repository's root app.js

