import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const packageDir = path.join(root, "paper_data_package");
const outputPath = path.join(packageDir, "timepix_paper_data_package.xlsx");

const csvFiles = [
  ["00_index", "00_experiment_index.csv"],
  ["01_main", "01_main_results_summary.csv"],
  ["02_runs", "02_run_level_results.csv"],
  ["03_per_class", "03_per_class_results.csv"],
  ["04_errors", "04_error_structure.csv"],
  ["05_modality", "05_modality_and_gate_diagnostics.csv"],
  ["06_features", "06_handcrafted_feature_results.csv"],
  ["07_loss", "07_loss_strategy_results.csv"],
  ["08_excluded", "08_excluded_or_diagnostic_runs.csv"],
];

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (ch === "\"") {
      if (inQuotes && next === "\"") {
        cell += "\"";
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      row.push(cell);
      cell = "";
    } else if ((ch === "\n" || ch === "\r") && !inQuotes) {
      if (ch === "\r" && next === "\n") i += 1;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += ch;
    }
  }
  if (cell.length > 0 || row.length > 0) {
    row.push(cell);
    rows.push(row);
  }
  return rows;
}

function coerceValue(value) {
  if (value === "") return null;
  if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(value)) {
    const num = Number(value);
    if (Number.isFinite(num)) return num;
  }
  return value;
}

function columnName(index) {
  let n = index + 1;
  let name = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    name = String.fromCharCode(65 + rem) + name;
    n = Math.floor((n - 1) / 26);
  }
  return name;
}

const workbook = Workbook.create();

for (const [sheetName, fileName] of csvFiles) {
  const csvText = await fs.readFile(path.join(packageDir, fileName), "utf8");
  const rows = parseCsv(csvText).map((row) => row.map(coerceValue));
  const width = Math.max(...rows.map((row) => row.length));
  const normalized = rows.map((row) => {
    const out = [...row];
    while (out.length < width) out.push(null);
    return out;
  });
  const sheet = workbook.worksheets.add(sheetName);
  const range = `A1:${columnName(width - 1)}${normalized.length}`;
  sheet.getRange(range).values = normalized;
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.inspect({
  kind: "table",
  range: "01_main!A1:K12",
  include: "values",
  tableMaxRows: 12,
  tableMaxCols: 11,
});
console.log(preview.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);
