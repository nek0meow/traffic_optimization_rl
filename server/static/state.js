export const state = {
    schema: null,
    histories: [],
    charts: {}
};

export function setSchema(schema) {
    state.schema = schema;
}

export function addHistory(modelName, history) {
    state.histories[modelName] = history;
}

export function removeHistory(modelName) {
    if (Object.hasOwn(state.histories, modelName)) {
        delete state.histories[modelName];
    }
}

export function addChart(key, chart) {
    state.charts[key] = chart
}

export function removeChart(key) {
    if (Object.hasOwn(state.charts, key)) {
        state.charts[key].destroy();
        delete state.charts[key];
    }
}