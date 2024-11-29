import requests  # HTTP 요청을 위한 라이브러리 
import logging   # 로깅 기능 제공 라이브러리

class SpellChecker:
    def __init__(self, api_key):
        # 로깅 시스템 초기화 (정보 레벨로 설정)
        logging.basicConfig(level=logging.INFO)
        
        # ML4K API 접근을 위한 키와 URL 설정
        self.key = api_key
        self.url = f"https://machinelearningforkids.co.uk/api/scratch/{self.key}/classify"
        
        # 맞춤법 교정을 위한 사전 생성
        self.correction_map = {
            # 기존 오타나 비표준 표현들
            "안되": "안 돼", "어의없다": "어이없다", 
            "급새": "금세", "웬만하면": "왠만하면", 
            "웬떡이야": "왠떡이야", "할께요": "할게요", 
            "않되나요": "안 되나요", "바램": "바람", 
            "역활": "역할", "내꺼": "내 것",
            
            # 추가된 맞춤법 교정 목록
            "괜차나요": "괜찮나요", 
            "아니요": "아니오", 
            "어케": "어떻게", 
            "어케됐어": "어떻게 됐어", 
            "됐어": "됐어", 
            "모르겟어": "모르겠어", 
            "몰랏어": "몰랐어"
        }

    def classify(self, text):
        """ML4K API로 텍스트 분류"""
        try:
            # API에 GET 요청 (타임아웃 5초)
            response = requests.get(self.url, params={"data": text}, timeout=5)
            # HTTP 오류 발생 시 예외 발생
            response.raise_for_status()
            # API 응답의 첫 번째 결과 반환
            return response.json()[0]
        except Exception as e:
            # 오류 발생 시 로깅하고 None 반환
            logging.error(f"ML4K API 호출 오류: {e}")
            return None

    def check_spelling(self, text):
        """맞춤법 검사 메서드"""
        # API로 텍스트 분류 결과 얻기
        result = self.classify(text)
        if result is None:
            # API 호출 실패 시 에러 상태 반환
            return {'status': 'error', 'message': 'API 호출 실패'}
        
        # 텍스트를 단어로 분리
        words = text.split()
        corrected_words = []  # 수정된 단어 저장 리스트
        corrections = []  # 교정 내역 저장 리스트
        
        # 각 단어에 대해 맞춤법 교정 수행
        for word in words:
            if word in self.correction_map:
                # 맞춤법 사전에 있으면 수정된 단어로 대체
                corrected_word = self.correction_map[word]
                corrected_words.append(corrected_word)
                corrections.append((word, corrected_word))
            else:
                # 사전에 없으면 원래 단어 유지
                corrected_words.append(word)
        
        # 수정된 단어들을 다시 문장으로 결합
        corrected_text = ' '.join(corrected_words)
        
        # 검사 결과 반환
        return {
            'original': text,  # 원본 텍스트
            'correction': corrected_text,  # 수정된 텍스트
            'confidence': result['confidence'],  # API 신뢰도
            'status': 'incorrect' if corrections else 'correct',  # 오류 상태
            'corrections': corrections,  # 교정 내역
            'ml4k_label': result["class_name"]  # ML4K 분류 라벨
        }

def main():
    # ML4K API 키 설정
    API_KEY = "be9502b0-959f-11ef-b8e1-6fc140b6f134750fc775-1168-4f68-9efc-678b9bf4ea68"
    
    # SpellChecker 인스턴스 생성
    checker = SpellChecker(API_KEY)
    
    while True:
        # 사용자로부터 텍스트 입력 받기
        text = input("\n맞춤법을 검사할 문장을 입력하세요 (종료: 'q'): ")
        
        # 'q' 입력 시 프로그램 종료
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
            print(f"신뢰도: {result['confidence']:.1f}%")
        else:
            print("맞춤법 오류:")
            for original, correction in result['corrections']:
                print(f"- '{original}' → '{correction}'")
            print(f"수정된 문장: {result['correction']}")
            print(f"신뢰도: {result['confidence']:.1f}%")

# 스크립트가 직접 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    main()
