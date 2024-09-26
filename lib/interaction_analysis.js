//npm_fladsync/lib/interaction_analysis.js

function analyzeInteraction(interactionType, content, duration) {
  let weight = 1.0;

  // 상호작용 유형에 따른 가중치 설정 (예시)
  if (interactionType === "article_click") {
    weight = 1.5;
  } else if (interactionType === "search") {
    weight = 1.2;
  }

  return { content, weight };
}

module.exports = { analyzeInteraction };
