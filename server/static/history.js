import { state, removeHistory, removeChart, addHistory, addChart } from "./state.js"

export async function toggleHistory() {
    const model = document.getElementById("history-select").value;

    if (state.histories[model]) {
        removeHistory(model);
        redrawCharts();
        return;
    }
    const response = await fetch(`/history/${model}`);
    const data = await response.json();

    if (data.error) {
        alert(data.error);
        return;
    }
    addHistory(model, data)
    redrawCharts();
}


function redrawCharts(data) {
    const avgWindow = parseInt(document.getElementById("avg-window").value)

    for (const key in state.charts) {
        removeChart(key)
    }

    const metrics = [
        ["avg_reward", "Средний reward"],
        ["avg_speed", "Средняя скорость, м/с"],
        ["avg_wait", "Среднее время ожидания, с"],
        ["avg_vehicles", "Среднее количество машин"]
    ];

    const chartsContainer =
        document.getElementById("charts-container");

    chartsContainer.innerHTML = "";

    for (const [metricKey, title] of metrics) {
        const wrapper = document.createElement("div");
        wrapper.className = "chart-wrapper";
        const canvas = document.createElement("canvas");
        wrapper.appendChild(canvas);
        chartsContainer.appendChild(wrapper);

        const datasets = []

        for (const modelName in state.histories) {
            const history = state.histories[modelName];
            const rawY = history[metricKey];
            const smoothY = movingAverage(rawY, avgWindow);
            const smoothX = history.episode.slice(avgWindow - 1);

            const points = smoothX.map((x, i) => ({
                x: x,
                y: smoothY[i]
            }));

            datasets.push({
                label: modelName,
                data: points,
                tension: 0.15,
                borderWidth: 0.6,
                pointRadius: 0.6
            })
        }

        const chart = new Chart(canvas, {
            type: "line",
            data: {
                datasets: datasets
            },

            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    legend: {
                        display: true
                    }
                },
                scales: {
                    x: {
                        type: "linear",
                        offset: false,
                        bounds: data,
                        ticks: {
                            autoskip: true
                        },

                        title: {
                            display: true,
                            text: "Эпизод"
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: title
                        }
                    }
                }
            }
        });

        addChart(metricKey, chart);
    }
}

function movingAverage(arr, windowSize) {
    if (arr.length < windowSize) {
        return arr;
    }
    const result = [];

    let temp_sum = 0
    for (let i = 0; i < windowSize; i++) {
        temp_sum += arr[i];
    }
    result.push(temp_sum / windowSize)
    for (let i = windowSize; i < arr.length; i++) {
        temp_sum -= arr[i - windowSize]
        temp_sum += arr[i]
        result.push(temp_sum / windowSize)
    }

    return result;
}