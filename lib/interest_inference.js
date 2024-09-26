const { extractFeatures } = require("./feature_extraction");

function inferInterest(text) {
  return new Promise((resolve, reject) => {
    console.log("Starting interest inference...");

    extractFeatures(text)
      .then((features) => {
        console.log(`Features extracted: ${features}`);
        // 여기서 실제 관심사 유추 로직을 수행
        resolve(features);
      })
      .catch((error) => {
        console.error(`Error during interest inference: ${error}`);
        reject(error);
      });
  });
}

module.exports = { inferInterest };
