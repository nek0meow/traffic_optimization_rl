import { setSchema } from "./state.js"
import { launchTraining, stopTraining } from "./training.js"
import { toggleHistory } from "./history.js"

async function loadConfigSchema() {
    const res = await fetch("/config-schema");
    const schema = await res.json();
    setSchema(schema)

    const trainSection = document.getElementById("train-sec");
    const envSection = document.getElementById("env-sec");

    populateSection("Training", schema.train, trainSection);
    populateSection("Environment", schema.env, envSection);
}

function populateSection(title, fields, container) {
    const h3 = document.createElement("h3");
    h3.innerText = title;
    container.appendChild(h3);

    for (const field of fields) {
        const label = document.createElement("label");
        label.innerText = field.name;

        const input = document.createElement("input");
        input.id = field.name;
        input.value = field.default;

        if (field.type === "int" || field.type === "float") {
            input.type = "number";
        } else {
            input.type = "text";
        }
        container.appendChild(label);
        container.appendChild(input);
    }
}

async function launchGui() {
    const map = document.getElementById("map-select").value;

    const modelPath =
        document.getElementById("model-select").value;

    await fetch("/launch-gui", {
        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            map: map,
            model_path: modelPath,
            greedy: true
        })
    });
}

async function updateLogs() {
    const response = await fetch("/logs");
    const logs = await response.json();
    const logsDiv = document.getElementById("logs");
    const wasNearBottom =
        logsDiv.scrollHeight - logsDiv.scrollTop - logsDiv.clientHeight < 50;

    logsDiv.innerHTML = logs.join("<br>");
    if (wasNearBottom) {
        logsDiv.scrollTop = logsDiv.scrollHeight;
    }
}

loadConfigSchema();
setInterval(updateLogs, 2000);

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('launch-button')?.addEventListener('click', launchTraining);
    document.getElementById('stop-button')?.addEventListener('click', stopTraining);
    document.getElementById('launch-gui-button')?.addEventListener('click', launchGui);
    document.getElementById('toggle-history-button')?.addEventListener('click', toggleHistory);
});