import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import requests

class IntegratedSpellChecker:
    def __init__(self, model_path):
        # Teachable Machine 모델 로드
        self.model = tf.keras.models.load_model(model_path)
        
        # ML4K API 키
        self.key = "be9502b0-959f-11ef-b8e1-6fc140b6f134750fc775-1168-4f68-9efc-678b9bf4ea68"
        self.url = f"https://machinelearningforkids.co.uk/api/scratch/{self.key}/classify"
        
        # 맞춤법 수정 사전
        self.correction_map = {
            "안되": "안 돼",
            "어의없다": "어이없다",
            "급새": "금세",
            "웬만하면": "왠만하면",
            "웬떡이야": "왠떡이야",
            "아따대고": "얼마 대고",
            "할께요": "할게요",
            "않되나요": "안 되나요",
            "바램": "바람",
            "잠깟다": "잠갔다",
            "오랫만에": "오랜만에",
            "역활": "역할",
            "나중에베요": "나중에 봐요",
            "건들이다": "건드리다",
            "애띠다": "엣되다",
            "설겆이": "설거지",
            "일일히": "일일이",
            "어떻해": "어떻게",
            "재작년": "재작년",
            "설래임": "설렘",
            "내꺼": "내 것",
            "멋일": "며칠",
            "단언컨대": "단언컨대",
            "되물럼": "되물림"
        }

    def classify_ml4k(self, text):
        """ML4K API를 사용하여 텍스트 분류"""
        try:
            response = requests.get(self.url, params={"data": text})
            
            if response.ok:
                response_data = response.json()
                top_match = response_data[0]
                return top_match
            else:
                response.raise_for_status()
        except Exception as e:
            print(f"ML4K API Error: {str(e)}")
            return None

    def preprocess_text_to_image(self, text):
        """텍스트를 이미지로 변환하여 Teachable Machine 모델 입력으로 사용"""
        img = np.ones((224, 224, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 0, 0)
        thickness = 2
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2
        
        cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, thickness)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img

    def check_with_teachable(self, text):
        """Teachable Machine 모델을 사용하여 텍스트 검사"""
        img = self.preprocess_text_to_image(text)
        prediction = self.model.predict(img)
        is_correct = np.argmax(prediction[0]) == 1
        confidence = float(np.max(prediction[0]))
        return is_correct, confidence

    def check_spelling(self, text):
        """통합된 맞춤법 검사 수행"""
        # ML4K API로 검사
        ml4k_result = self.classify_ml4k(text)
        
        if ml4k_result is None:
            return {
                'original': text,
                'correction': None,
                'confidence': 0,
                'status': 'error'
            }

        # Teachable Machine으로 검사
        tm_is_correct, tm_confidence = self.check_with_teachable(text)
        
        # 결과 통합 (ML4K와 Teachable Machine 결과의 평균)
        combined_confidence = (ml4k_result["confidence"] + tm_confidence * 100) / 2
        
        # 텍스트를 단어로 분리하고 각각 검사
        words = text.split()
        corrected_words = []
        corrections = []
        
        for word in words:
            if word in self.correction_map:
                corrected_word = self.correction_map[word]
                corrected_words.append(corrected_word)
                corrections.append((word, corrected_word))
            else:
                corrected_words.append(word)
        
        corrected_text = ' '.join(corrected_words)
        
        return {
            'original': text,
            'correction': corrected_text,
            'confidence': combined_confidence,
            'status': 'incorrect' if corrections else 'correct',
            'corrections': corrections,
            'ml4k_label': ml4k_result["class_name"],
            'teachable_correct': tm_is_correct
        }

def main():
    # Teachable Machine 모델 경로 설정
    model_path = 'path/to/your/teachable_machine_model.h5'
    
    # 맞춤법 검사기 초기화
    checker = IntegratedSpellChecker(model_path)
    
    while True:
        # 사용자 입력 받기
        text = input("\n검사할 텍스트를 입력하세요 (종료하려면 'q' 입력): ")
        
        if text.lower() == 'q':
            print("프로그램을 종료합니다.")
            break
        
        # 맞춤법 검사 수행
        result = checker.check_spelling(text)
        
        # 결과 출력
        if result['status'] == 'error':
            print("검사 중 오류가 발생했습니다.")
        elif result['status'] == 'correct':
            print(f"'{text}'는 올바른 맞춤법입니다.")
            print(f"ML4K 결과: {result['ml4k_label']}")
            print(f"종합 신뢰도: {result['confidence']:.1f}%")
        else:
            print("틀린 맞춤법이 발견되었습니다:")
            for original, correction in result['corrections']:
                print(f"- '{original}' -> '{correction}'")
            print(f"수정된 문장: '{result['correction']}'")
            print(f"ML4K 결과: {result['ml4k_label']}")
            print(f"종합 신뢰도: {result['confidence']:.1f}%")

if __name__ == "__main__":
    main()
