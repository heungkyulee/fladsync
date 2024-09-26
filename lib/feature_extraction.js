const { spawn } = require("child_process");
const path = require("path");

function extractFeatures(text) {
  return new Promise((resolve, reject) => {
    console.log("Starting feature extraction...");

    const scriptPath = path.resolve(__dirname, "../models/inference_model.py");
    console.log(`Python script path: ${scriptPath}`);

    const process = spawn("python3", [scriptPath, text]);

    let resultData = "";

    process.stdout.on("data", (data) => {
      resultData += data.toString(); // 데이터를 누적하여 저장
    });

    process.stdout.on("end", () => {
      console.log(`Received data from Python script: ${resultData}`);
      try {
        // Python에서 올바르게 JSON을 출력하므로, JSON.parse를 사용
        const parsedData = JSON.parse(resultData);
        resolve(parsedData);
      } catch (error) {
        reject(`Error parsing data: ${error}`);
      }
    });

    process.stderr.on("data", (data) => {
      console.error(`Error from Python script: ${data}`);
      reject(`Error: ${data}`);
    });

    process.on("close", (code) => {
      if (code !== 0) {
        console.error(`Python process exited with code ${code}`);
        reject(`Python process exited with code ${code}`);
      } else {
        console.log("Python process finished successfully.");
      }
    });
  });
}

module.exports = { extractFeatures };
