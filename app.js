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

const patternData = {
  "Single Episode with Recovery": {
    points: [12, 18, 32, 56, 78, 72, 58, 42, 28, 18, 14, 12],
    description: "One distinct episode rises, peaks, and gradually settles back toward baseline after treatment starts.",
  },
  "Recurrent Depression": {
    points: [18, 24, 62, 54, 24, 22, 25, 68, 58, 26, 22, 20],
    description: "Symptoms improve and then return in later episodes, which is common after a previous depressive episode.",
  },
  "Chronic Depression": {
    points: [58, 60, 64, 61, 57, 63, 66, 60, 56, 62, 64, 59],
    description: "Symptoms stay present over a long period with modest ups and downs rather than clear recovery periods.",
  },
  "Treatment Response": {
    points: [82, 79, 71, 63, 54, 44, 35, 28, 24, 20, 18, 16],
    description: "Consistent treatment can produce a progressive reduction in symptom burden over weeks rather than days.",
  },
};

const treatmentData = {
  "Major Depression": {
    treatments: [["Combined Med+CBT", 78], ["ECT", 80], ["SSRIs", 65], ["SNRIs", 63], ["CBT", 58]],
    notes: [
      "Combined medication and CBT usually produces the best overall response.",
      "ECT is often reserved for severe or treatment-resistant cases.",
      "Medication response commonly starts within a few weeks, while therapy builds skills over time.",
    ],
  },
  "Treatment-Resistant Depression": {
    treatments: [["ECT", 70], ["Ketamine", 60], ["Esketamine", 57], ["TMS", 45], ["CBT+Medication", 40]],
    notes: [
      "Treatment-resistant depression often needs escalation beyond a standard antidepressant trial.",
      "Neuromodulation and ketamine-based treatments are typically specialist pathways.",
      "Medication plus therapy remains a strong combination even in harder cases.",
    ],
  },
  "Depression with Anxiety": {
    treatments: [["Combined Med+CBT", 80], ["SNRIs", 72], ["SSRIs", 70], ["CBT for Anxiety", 65], ["Mindfulness", 55]],
    notes: [
      "SSRIs and SNRIs are often the first medication choices when anxiety is prominent.",
      "CBT tailored to both depression and anxiety improves the overall fit of treatment.",
      "Mindfulness can be useful as an adjunct rather than a full substitute for treatment.",
    ],
  },
  "Bipolar Depression": {
    treatments: [["Olanzapine+Fluoxetine", 65], ["Quetiapine", 60], ["Lurasidone", 55], ["Lamotrigine", 50], ["Psychotherapy", 45]],
    notes: [
      "Bipolar depression is managed differently from unipolar depression.",
      "Antidepressant monotherapy is usually avoided because of switching risk.",
      "Mood stabilizer-based treatment is the core strategy.",
    ],
  },
};

const tabButtons = document.querySelectorAll(".tab-button");
const panels = document.querySelectorAll(".panel");
const quickPrompts = document.querySelectorAll(".prompt-chip");
const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatStatus = document.getElementById("chatStatus");
const sourceList = document.getElementById("sourceList");
const symptomInput = document.getElementById("symptomInput");
const predictButton = document.getElementById("predictButton");
const predictStatus = document.getElementById("predictStatus");
const predictionResults = document.getElementById("predictionResults");
const phq9Form = document.getElementById("phq9Form");
const phq9Result = document.getElementById("phq9Result");
const phq9History = document.getElementById("phq9History");
const patternSelect = document.getElementById("patternSelect");
const patternChart = document.getElementById("patternChart");
const patternDescription = document.getElementById("patternDescription");
const treatmentSelect = document.getElementById("treatmentSelect");
const treatmentChart = document.getElementById("treatmentChart");
const treatmentNotes = document.getElementById("treatmentNotes");
const riskButton = document.getElementById("riskButton");
const riskForm = document.getElementById("riskForm");
const riskResult = document.getElementById("riskResult");

function switchTab(tabId) {
  tabButtons.forEach((button) => button.classList.toggle("active", button.dataset.tab === tabId));
  panels.forEach((panel) => panel.classList.toggle("active", panel.id === `panel-${tabId}`));
}

tabButtons.forEach((button) => button.addEventListener("click", () => switchTab(button.dataset.tab)));

function addMessage(role, text) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = text;
  chatMessages.appendChild(node);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function renderSources(sources = []) {
  sourceList.innerHTML = "";
  if (!sources.length) {
    sourceList.innerHTML = '<div class="empty-state">Sources for the latest answer will appear here.</div>';
    return;
  }

  sources.forEach((source, index) => {
    const item = document.createElement("div");
    item.className = "source-item";
    item.textContent = `${index + 1}. ${source}`;
    sourceList.appendChild(item);
  });
}

async function sendChatPrompt(prompt) {
  const query = prompt.trim();
  if (!query) return;

  addMessage("user", query);
  chatInput.value = "";
  chatStatus.textContent = "Thinking...";

  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (!response.ok) throw new Error(`Request failed with ${response.status}`);

    const payload = await response.json();
    addMessage("bot", payload.answer || "No answer returned.");
    renderSources(payload.sources || []);
    chatStatus.textContent = "Ready";
  } catch (error) {
    addMessage("bot", "The chat service is unavailable right now. Please try again.");
    chatStatus.textContent = "Request failed";
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendChatPrompt(chatInput.value);
});

quickPrompts.forEach((button) => button.addEventListener("click", () => sendChatPrompt(button.dataset.prompt || "")));

function renderPredictionCard(item) {
  const probability = Math.round((item.probability || 0) * 100);
  return `
    <div class="prediction-card">
      <h4>${item.disease}</h4>
      <div class="meter"><span style="width:${probability}%"></span></div>
      <p><strong>${probability}%</strong> likelihood in this lightweight model.</p>
      <p>${item.info?.description || ""}</p>
      <p><strong>Common symptoms:</strong> ${(item.info?.symptoms || []).join(", ")}</p>
      <p><strong>Typical treatments:</strong> ${(item.info?.common_treatments || []).join(", ")}</p>
    </div>
  `;
}

predictButton.addEventListener("click", async () => {
  const symptoms = symptomInput.value.trim();
  if (!symptoms) return;

  predictStatus.textContent = "Scoring symptom patterns...";
  predictionResults.innerHTML = "";

  try {
    const response = await fetch(`${API_BASE}/disease-predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symptoms }),
    });
    if (!response.ok) throw new Error(`Request failed with ${response.status}`);

    const payload = await response.json();
    predictionResults.innerHTML = (payload.predictions || []).map(renderPredictionCard).join("");
    predictStatus.textContent = "Finished";
  } catch (error) {
    predictionResults.textContent = "Prediction service is unavailable right now.";
    predictStatus.textContent = "Request failed";
  }
});

function buildPhqForm() {
  phqQuestions.forEach((question, index) => {
    const wrapper = document.createElement("div");
    wrapper.className = "question-card";
    wrapper.innerHTML = `<div>${index + 1}. ${question}</div>`;

    const options = document.createElement("div");
    options.className = "question-options";
    phqOptions.forEach(([label, value]) => {
      const optionLabel = document.createElement("label");
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = `phq-${index}`;
      radio.value = value;
      if (value === 0) radio.checked = true;
      optionLabel.append(radio, document.createTextNode(label));
      options.appendChild(optionLabel);
    });
    wrapper.appendChild(options);
    phq9Form.appendChild(wrapper);
  });
}

function interpretPhq(score) {
  if (score <= 4) return ["Minimal or none", "You have few or no depressive symptoms right now."];
  if (score <= 9) return ["Mild", "Symptoms may be present but are often manageable with support and monitoring."];
  if (score <= 14) return ["Moderate", "Symptoms are likely affecting daily life and deserve closer attention."];
  if (score <= 19) return ["Moderately severe", "Symptoms may be significantly disruptive and professional evaluation is advisable."];
  return ["Severe", "Symptoms appear severe and prompt professional support is strongly advisable."];
}

function phqRecommendations(score) {
  const base = [
    "Keep a consistent sleep routine.",
    "Aim for regular physical activity most days of the week.",
    "Reduce isolation and stay connected with trusted people.",
  ];
  if (score <= 4) return [...base, "Monitor for changes if your mood worsens over time."];
  if (score <= 9) return [...base, "Consider self-help tools or talking to a clinician if symptoms persist for more than two weeks."];
  if (score <= 14) return [...base, "A structured conversation with a healthcare professional would be reasonable at this stage."];
  if (score <= 19) return [...base, "Professional care is strongly recommended. Therapy and medication may both be discussed."];
  return [...base, "If you may harm yourself, contact local emergency services or a crisis line immediately."];
}

function loadPhqHistory() {
  try { return JSON.parse(localStorage.getItem("medibot-phq-history") || "[]"); } catch { return []; }
}

function savePhqHistory(history) {
  localStorage.setItem("medibot-phq-history", JSON.stringify(history));
}

function renderPhqHistory() {
  const history = loadPhqHistory();
  phq9History.innerHTML = history.length ? "" : '<div class="empty-state">No local history yet.</div>';
  history.slice(-6).reverse().forEach((item) => {
    const row = document.createElement("div");
    row.className = "history-item history-bar";
    row.innerHTML = `<span>${item.date}</span><div class="meter"><span style="width:${(item.score / 27) * 100}%"></span></div><strong>${item.score}</strong>`;
    phq9History.appendChild(row);
  });
}

document.getElementById("phq9Submit").addEventListener("click", () => {
  const score = phqQuestions.reduce((total, _, index) => total + Number(document.querySelector(`input[name="phq-${index}"]:checked`)?.value || 0), 0);
  const [severity, interpretation] = interpretPhq(score);
  const recommendations = phqRecommendations(score);
  const date = new Date().toISOString().slice(0, 10);
  const history = loadPhqHistory();
  history.push({ date, score });
  savePhqHistory(history.slice(-10));
  renderPhqHistory();

  phq9Result.innerHTML = `
    <div class="score-badge">Score ${score} / 27 · ${severity}</div>
    <p>${interpretation}</p>
    <div class="meter"><span style="width:${(score / 27) * 100}%"></span></div>
    <ul>${recommendations.map((item) => `<li>${item}</li>`).join("")}</ul>
  `;
});

document.getElementById("phq9Reset").addEventListener("click", () => {
  phq9Form.reset();
  phq9Result.textContent = "Questionnaire reset.";
});

function calculateRisk(inputs) {
  let riskScore = 0;
  const factors = {};
  if (inputs.gender === "Female") { riskScore += 10; factors["Female gender"] = 10; }
  if (inputs.age_group === "18-29" || inputs.age_group === "60+") { riskScore += 10; factors[`Age ${inputs.age_group}`] = 10; }
  else if (inputs.age_group === "30-44") { riskScore += 5; factors[`Age ${inputs.age_group}`] = 5; }
  else { riskScore += 3; factors[`Age ${inputs.age_group}`] = 3; }
  if (inputs.family_history === "Yes") { riskScore += 15; factors["Family history"] = 15; }
  if (inputs.chronic_illness === "Yes") { riskScore += 12; factors["Chronic illness"] = 12; }
  if (inputs.recent_trauma === "Yes") { riskScore += 20; factors["Recent stressful event"] = 20; }
  if (inputs.social_support === "Low") { riskScore += 15; factors["Low social support"] = 15; }
  else if (inputs.social_support === "Medium") { riskScore += 5; factors["Medium social support"] = 5; }
  else { riskScore -= 5; factors["High social support"] = -5; }
  return { riskScore: Math.max(0, Math.min(100, riskScore)), factors };
}

function riskRecommendations(score, inputs) {
  const notes = [
    "Regular exercise, consistent sleep, and social connection lower baseline risk.",
    "Persistent changes in mood, drive, or functioning deserve clinical attention.",
  ];
  if (score >= 50) notes.push("This pattern suggests elevated risk. A mental health screening with a clinician would be reasonable.");
  else if (score >= 20) notes.push("This is a moderate-risk profile. Monitoring mood and stress load would be useful.");
  else notes.push("Current risk looks relatively low, but stressors can still change the picture.");
  if (inputs.recent_trauma === "Yes") notes.push("Recent stress is a major contributor. Give recovery, support, and structure extra attention.");
  if (inputs.social_support === "Low") notes.push("Low social support stands out here. Deliberately building support can materially reduce risk.");
  return notes;
}

riskButton.addEventListener("click", () => {
  const inputs = Object.fromEntries(new FormData(riskForm).entries());
  const { riskScore, factors } = calculateRisk(inputs);
  const notes = riskRecommendations(riskScore, inputs);
  const sortedFactors = Object.entries(factors).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  riskResult.innerHTML = `
    <div class="score-badge">Risk score ${riskScore.toFixed(0)} / 100</div>
    <div class="meter"><span style="width:${riskScore}%"></span></div>
    <div class="stack-list">
      ${sortedFactors.map(([name, value]) => `<div class="risk-metric"><strong>${name}</strong><div class="meter"><span style="width:${Math.min(Math.abs(value) * 4, 100)}%"></span></div><span>${value >= 0 ? "+" : ""}${value.toFixed(0)}</span></div>`).join("")}
    </div>
    <ul>${notes.map((item) => `<li>${item}</li>`).join("")}</ul>
  `;
});

function renderPatternChart(name) {
  const data = patternData[name];
  const width = 520;
  const height = 200;
  const padding = 18;
  const step = (width - padding * 2) / (data.points.length - 1);
  const coords = data.points.map((point, index) => {
    const x = padding + index * step;
    const y = height - padding - (point / 100) * (height - padding * 2);
    return [x, y];
  });
  const line = coords.map(([x, y]) => `${x},${y}`).join(" ");
  const area = `${line} ${width - padding},${height - padding} ${padding},${height - padding}`;
  patternChart.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${name} chart"><polygon class="chart-area" points="${area}"></polygon><polyline class="chart-line" points="${line}"></polyline></svg>`;
  patternDescription.textContent = data.description;
}

function renderTreatmentChart(name) {
  const data = treatmentData[name];
  treatmentChart.innerHTML = data.treatments.map(([label, value]) => `
    <div class="bar-row">
      <span>${label}</span>
      <div class="bar-track"><span style="width:${value}%"></span></div>
      <strong>${value}%</strong>
    </div>
  `).join("");
  treatmentNotes.innerHTML = `<ul>${data.notes.map((item) => `<li>${item}</li>`).join("")}</ul>`;
}

Object.keys(patternData).forEach((name) => patternSelect.add(new Option(name, name)));
Object.keys(treatmentData).forEach((name) => treatmentSelect.add(new Option(name, name)));

patternSelect.addEventListener("change", () => renderPatternChart(patternSelect.value));
treatmentSelect.addEventListener("change", () => renderTreatmentChart(treatmentSelect.value));

buildPhqForm();
renderPhqHistory();
renderSources([]);
renderPatternChart(Object.keys(patternData)[0]);
renderTreatmentChart(Object.keys(treatmentData)[0]);
addMessage("bot", "Ask a question about a condition, symptom, cause, or treatment to start.");
