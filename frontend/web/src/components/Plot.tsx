// react-plotly.js wired to the lightweight dist bundle (avoids the full plotly.js build).
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

import type { Layout } from "plotly.js";

export const Plot = createPlotlyComponent(Plotly);

/** Dark, transparent base layout matching the Mantine dark theme. */
export const baseLayout: Partial<Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#c1c2c5", size: 12 },
  margin: { l: 60, r: 20, t: 30, b: 50 },
  xaxis: { gridcolor: "rgba(255,255,255,0.08)", zerolinecolor: "rgba(255,255,255,0.15)" },
  yaxis: { gridcolor: "rgba(255,255,255,0.08)", zerolinecolor: "rgba(255,255,255,0.15)" },
  legend: { orientation: "h", y: -0.2 },
};

export const plotConfig = { displayModeBar: false, responsive: true } as const;
