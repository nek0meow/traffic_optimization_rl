import { state } from "./state.js"

export async function launchTraining() {
    const launchButton = document.getElementById("launch-button");
    const stopButton = document.getElementById("stop-button");

    const schema = state.schema

    const config = {
        model_name: document.getElementById("model-name").value,
        map: document.getElementById("map-select").value,

        train: {},
        env: {}
    };

    for (const field of schema.train) {
        const el = document.getElementById(field.name);

        config.train[field.name] = (() => {
            switch (el.type) {
                case "int":
                    return parseInt(el.value);
                case "float":
                    return parseFloat(el.value);
                default:
                    return el.value
            }
        })()
    }

    for (const field of schema.env) {
        const el = document.getElementById(field.name);

        config.env[field.name] = (() => {
            switch (el.type) {
                case "int":
                    return parseInt(el.value);
                case "float":
                    return parseFloat(el.value);
                default:
                    return el.value
            }
        })()
    }

    const res = await fetch("/launch", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(config)
    });

    const data = await res.json();
    if (data.error) {
        alert(data.error);
        return;
    } else {
        launchButton.disabled = true;
        stopButton.disabled = false;
    }
}


export async function stopTraining() {
    const launchButton = document.getElementById("launch-button");
    const stopButton = document.getElementById("stop-button");

    await fetch("/stop", {
        method: "POST"
    });

    launchButton.disabled = false;
    stopButton.disabled = true;
}