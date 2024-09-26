const { inferInterest } = require("./lib/interest_inference");
const { analyzeInteraction } = require("./lib/interaction_analysis");

function processInteraction(interactionType, content, duration) {
  return new Promise((resolve, reject) => {
    console.log(
      `Processing interaction: ${interactionType} - Content: ${content} - Duration: ${duration}`
    );

    const interaction = analyzeInteraction(interactionType, content, duration);
    console.log(`Interaction analyzed: ${JSON.stringify(interaction)}`);

    inferInterest(interaction.content)
      .then((result) => {
        console.log(`Inferred Interests: ${result}`);
        resolve(result);
      })
      .catch((error) => {
        console.error(`Error during processInteraction: ${error}`);
        reject(error);
      });
  });
}

module.exports = { processInteraction };
