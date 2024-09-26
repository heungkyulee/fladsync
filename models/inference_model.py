# import json
# import torch
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# import numpy as np

# class InterestInferenceModel:
#     def __init__(self):
#         self.tokenizer = DistilBertTokenizer.from_pretrained('saved_model/')
#         self.model = DistilBertForSequenceClassification.from_pretrained('saved_model/').to('cpu')
#         self.mlb_classes = np.load("mlb_classes.npy", allow_pickle=True)

#     def extract_features(self, text):
#         inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
#         outputs = self.model(**inputs)
#         logits = outputs.logits
#         predictions = torch.sigmoid(logits)
#         predicted_labels = predictions > 0.5
#         return [self.mlb_classes[i] for i, pred in enumerate(predicted_labels[0]) if pred]

# # Inference usage
# if __name__ == "__main__":
#     import sys
#     text = sys.argv[1]  # Node.js에서 넘긴 첫 번째 인자
#     model = InterestInferenceModel()
#     categories = model.extract_features(text)
#     print(json.dumps(categories))


import json
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 예시 주제 분류 레이블
LABELS = ["technology", "economy", "health", "education", "politics"]

class InterestInferenceModel:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = self.build_classifier()

    def extract_features(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def build_classifier(self):
        # Logistic Regression을 사용한 간단한 주제 분류기
        # 실제로는 벡터에 맞는 학습된 분류 모델이 필요합니다.
        classifier = make_pipeline(StandardScaler(), LogisticRegression())
        
        # 예시 학습 데이터 (벡터 -> 주제 레이블)
        # 실제 데이터로 학습시키거나 학습된 모델을 불러와야 합니다.
        X_train = np.random.rand(100, 768)  # 768차원의 임의 벡터
        y_train = np.random.randint(0, len(LABELS), size=100)  # 임의 레이블
        
        classifier.fit(X_train, y_train)
        return classifier

    def predict_topic(self, features):
        topic_index = self.classifier.predict(features)[0]
        return LABELS[topic_index]

# 명령행 인자로 받은 텍스트를 처리
if __name__ == "__main__":
    import sys
    text = sys.argv[1]  # Node.js에서 넘긴 첫 번째 인자
    model = InterestInferenceModel()
    features = model.extract_features(text)
    
    # 주제 예측
    topic = model.predict_topic(features)
    
    # 예측 결과 출력 (JSON 형식)
    result = {
        "text": text,
        "features": features.tolist(),
        "predicted_topic": topic
    }
    
    print(json.dumps(result))
