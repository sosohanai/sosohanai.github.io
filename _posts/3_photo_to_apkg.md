# 구글포토 이미지 OCR 적용후 문제유형
1. 구글포토 앨범이름으로 이미지 읽기
2. 구글 vision API 로 이미지 OCR 적용
3. openAI API로 문제, 보기 유형으로 변환
4. 변환된 파일로 apkg 생성


# 구글 vision API 인증키 json 파일 다운로드
1. google vision API 사용설정
구글클라우드 콘솔에서 vision API 사용설정후 인증키 json 파일 다운로드
현재 client_secret_desktop.json 파일로 사용
2. google photo 사용설정
구글포토 인증키 service-account-file.json 파일로 사용


```python
%pip install --upgrade google-api-python-client
%pip install --upgrade google-cloud-vision
%pip install --upgrade google-auth-oauthlib
```

### 기출이라는 앨범에서 이미지 OCR 적용후 문제유형으로 변환



```python
import html
import json
import os
import requests
from io import BytesIO
from google.cloud import vision
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import genanki
from openai import OpenAI

# OpenAI 클라이언트 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
open_ai_model = 'gpt-3.5-turbo'

# OAuth 2.0 클라이언트 ID와 비밀 정보 설정
client_secrets_file = 'client_secret_desktop.json'

# 스코프 설정
scopes = ['https://www.googleapis.com/auth/photoslibrary.readonly', 'https://www.googleapis.com/auth/cloud-vision']

# 사용자 인증 흐름 설정
flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes=scopes)

# 사용자 인증 및 액세스 토큰 획득
credentials = flow.run_local_server(port=8088)

# Google Photos API 서비스 객체 생성
service = build('photoslibrary', 'v1', credentials=credentials, static_discovery=False)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service-account-file.json'
# Google Cloud Vision API 서비스 객체 생성
client_vision = vision.ImageAnnotatorClient(client_options={'api_endpoint': 'us-vision.googleapis.com'})

# 정답 리스트
answer_list = [
    '①', '④', '②', '③', '③',
    '①', '②', '④', '②', '①',
    '④', '③', '③', '③', '③',
    '②', '①', '④', '④', '②',
    '②', '③', '③', '③', '②',
    '①', '③', '①', '②', '③',
    '①', '③', '③', '②', '①',
    '①', '④', '①', '③', '②',
    '③', '④', '③', '②', '①',
    '①', '②', '①', '③', '①',
    '①', '②', '③', '①', '③',
    '①', '①', '②', '③', '②',
    '①', '④', '③', '②', '②',
    '②', '④', '②', '③', '④',
    '①', '①', '②', '③', '①',
    '④', '④', '②', '②', '①'
]


def get_photo_urls(examination):
    """앨범에서 사진의 URL 가져오는 함수"""
    # us-vision.googleapis.com

    response = service.mediaItems().search(body={'albumId': examination}).execute()
    items = response.get('mediaItems', [])
    return [item['baseUrl'] for item in items]


def annotate_image(image_url):
    """이미지를 Vision API 사용하여 주석  처리하는 함수"""
    # 이미지 너비를 원본크기로 설정하고, 높이를 원본 크기로 설정하며, 이미지를 잘라내지 않고 압축하지 않으며 오버레이를 표시하지 않습니다.
    original_image = '=w0-h0-n-k-no'
    response = requests.get(image_url + original_image, stream=True)
    if response.status_code == 200:
        image_bytes = BytesIO(response.content)
        image_byte_array = image_bytes.getvalue()
        content = {'content': image_byte_array}
        image = vision.Image(**content)
        # image.show()
        features = [{"type_": vision.Feature.Type.DOCUMENT_TEXT_DETECTION}]
        return client_vision.annotate_image({"image": image, "features": features}).full_text_annotation


def convert_ocr_to_text(document):
    text = ""

    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    text += "".join([symbol.text for symbol in word.symbols])
                    text += " "
                text += "\n"
    return text.strip()


# OCR 파이프라인 실행 함수
def run_ocr_pipeline(album_title, deck_title, apkg_title):
    response = service.albums().list().execute()

    # 앨범 목록 가져오기
    albums = response.get('albums', [])

    # 리스트 컴프리헨션을 사용하여 특정 앨범의 ID 찾기
    album_id = next((album['id'] for album in albums if album['title'] == album_title), None)

    # """OCR 파이프라인 실행 함수"""
    photo_urls = get_photo_urls(album_id)
    for idx, url in enumerate(photo_urls, start=1):
        # if idx == 3: # 테스트용 3개만
            print(f'Processing image {idx}/{len(photo_urls)}')
            document = annotate_image(url)
            ocr_text = convert_ocr_to_text(document)
            file_name = apkg_title + str(idx) + '.apkg'
            write_to_file(file_name, generate_anki_deck(ocr_text, deck_title))


def generate_anki_deck(ocr_texts, name):
    """Anki 덱을 생성하는 함수"""
    # 덱 지정
    my_deck = genanki.Deck(12347778, name)
    # 모델 지정
    anki_my_model = genanki.Model(
        109123411,
        '객관식 정답과 해설 모델',
        fields=[
            {'name': 'question_no'},
            {'name': 'question'},
            {'name': 'choices'},
            {'name': 'answer'}
        ],
        templates=[
            {'name': 'Card {{question_no}}',
             'qfmt': '{{question_no}}. {{question}}<br><br>{{choices}}',
             'afmt': '{{FrontSide}}<hr id="answer">{{answer}}'}
        ]
    )
    response_open = client.chat.completions.create(
        messages=[
            {"role": "user", "content": "I have the following text"},
            {"role": "user", "content": ocr_texts},
            {"role": "user", "content":
                "이게 문제목록이야. 두자리 숫자로 시작되는 문제가 있고 question_no 필드로 문자형태로 되어있어. 앞 문제번호와 그다음 문제번호는 1씩 차이나. "
                "예를 들어 question_no가 '07'이면 그 앞문제 번호는 '06'이 되 그 뒤문제 번호는 '08이야. '594'는 '68'번 문제야"
                "문제는 4~6개씩 있어. 문제번호가 없는 경우 앞뒤 상황으로 유추해줘. 문제는 빅데이터분석기사의 내용에 맞는건지 확인해줘"
                "①과 같은 형태로 된건 보기야. 문제번호인 question_no는 문자열로 바뀐형태야."
                "배열 형태인 choices 가 있고 choices 는 보기4개로 이뤄진 list. 각 list 는 숫자.보기 내용 텍스트로 되어있어."
                "각 choices 에 ①과 같은 숫자기호가 없는 choices 들은 유추해서 넣어. "
                "숫자기호는 1,2,3,4로 대체해서 넣어. 모든 숫자기호는 ①과 같은 기호가 아닌 숫자로 대체해"
                "choices 의 choice 는 각각 한줄이 되게 띄어쓰기를 붙여줘. choices 는 4개의 보기로 이뤄진 list."
                "choices 의  첫번째 보기는 1.로 시작하고, 두번째 보기는 2.로 시작하고 세번째 보기는 3.으로 시작하고 네번째 보기는 4.으로 시작해."
                "보기내용이 없으면 빅데이터분석기사의 내용으로 문제에 맞는 보기를 유추해서  < 는 &lt;로 바꾸고 > 는 &gt;로 바꿔서 넣어줘"
                "문제와 문제사이가 보기니까 다음문제 앞에 ①과 같은 보기영역을 찾아서 4로 나누어 잘 매핑해줘."
                "다음의 의사 결정 나무에서 x값을 구하시오.와 같은 문제의 경우 51번 문제 윗줄들 보기 4줄만 처리해줘. 문제와 보기사이의 내용들은 문제내용으로 문제쪽으로 붙여줘."
                " < 또는 > 같은건 escape 처리해줘"
                "모든 문제, 보기값들, 답값은 escape 처리해줘. "
                "< 는 &lt;로 바꾸고 > 는 &gt;로 바꿔줘. 예를 들어 1. x < 10는 1. x &lt; 10으로 바꿔줘. 꼭 html escape 처리를 해줘"
                "questions 는 question_no, question, choices, answer,explanation 5개 필드를 가진 object 이고 json."
                "만약 json 되기 어려우면 question_no는 유추한 그대로, question 이 없으면 문제깨짐 이라고 넣고 json 맞춰줘."
                "answer 는 question_no에 맞는 정답번호를 찾아 answer_list 매핑해준 뒤에 추가로 한칸 띄우고 정답에 맞는 보기내용도 같이 매핑해줘"
                "explanation 필드는 문제와 보기에 대한 해설도 빅데이터분석기사내용을 토대로 자세히 넣어줘"
                + str(anki_my_model) + "의 구조에 맞게 "
               "각각 question_no, question, choices, answer,explanation 5개 필드를 가진 object 되어있어"
               "코드블럭표시하는 ```json ```제거하고 줘. 설명도 제거하고 questions 필드를 가진 json 만들어줘. json.loads()로 파싱가능한 형태로 줘."
               "꼭  questions 필드를 가진 object 여야해. 중간 생략없이 줘"
             }
        ],
        model=open_ai_model
    )
    result_data = response_open.choices[0].message.content
    if result_data.startswith('```json'):
        result_data = result_data.replace('```json', '').replace('```', '')
    json_object = {}
    
    try:
        json_object = json.loads(result_data)
        print('json.loads')
        print(json_object)
    except json.JSONDecodeError:
        print('JSON 형식이 아닙니다.')
        print(result_data)
        

    try:
        json_object.get('questions')
    except AttributeError:
        print('questions 속성이 없습니다.')
        print(result_data)

    for question_data in json_object.get('questions', []):
        if question_data['choices'] != '':
            question = html.escape(question_data['question'])
            choices = html.escape("".join(question_data['choices']))
            answer = f"정답: {html.escape(question_data['answer'])} 해설: {html.escape(question_data['explanation'])}"
            note = genanki.Note(
                model=anki_my_model,
                fields=[question_data['question_no'], question, choices, answer]
            )
            my_deck.add_note(note)
    return my_deck


def write_to_file(file_name, my_deck):
    """Anki 패키지 파일을 생성하는 함수"""
    genanki.Package(my_deck).write_to_file(file_name)
    print(f'{file_name} 파일이 생성되었습니다.')


# 코드 실행############################################################################################################
if __name__ == "__main__":
    # 앨범 지정
    album_name = '기출'
    # 덱 버전 지정
    deck_version = 'VER1'
    # 덱 이름 지정
    deck_name = album_name + '_' + deck_version + '_DECK'
    # 패키지 이름 지정
    apkg_name = deck_name
    run_ocr_pipeline(album_name, deck_name, apkg_name)

```

    Please visit this URL to authorize this application: https://accounts.google.com/o/oauth2/auth?response_type=code&client_id=321040457013-9c5098pkpmsdma3c2psajfong2kdv1kq.apps.googleusercontent.com&redirect_uri=http%3A%2F%2Flocalhost%3A8088%2F&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fphotoslibrary.readonly+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcloud-vision&state=bGZHElhRiQrUrIK8EUcPHATSnBETwh&access_type=offline
    Processing image 1/17
    json.loads
    {'questions': [{'question_no': '06', 'question': '다음 중 데이터 분석 수준 진단 결과에 대한 설명으로 옳지 않은 것은?', 'choices': ['1. 정착형은 준비도는 높으나 조직, 인력, 분석 업무, 분석 기법 등을 기업 내부에서 제한적으로 사용하는 경우이다.', '2. 준비형은 기업에 필요한 데이터, 인력, 조직, 분석 업무, 분석 기법 등이 적용되어 있지 않아 사전비가 필요한 경우이다.', '3. 도입형은 기업에서 활용하는 분석 업무, 기법 등이 부족하지만 적용조직 등 준비도가 높아 바로 도입할 수 있는 경우이다.', '4. 확산형은 기업에 필요한 6가지 분석 구성요소를 갖추고 있고, 현재 부분적으로 도입되어 지속적인 확산이 필요한 경우이다.'], 'answer': '2', 'explanation': '정착형, 준비형, 도입형, 확산형은 데이터 분석 수준 진단 결과에 대한 설명으로 정확하게 설명된 것이다.'}, {'question_no': '07', 'question': '다음 중 분석준비도(Readiness)의 진단 영역으로 옳지 않은 것은?', 'choices': ['1. 분석문화', '2. 분석결과', '3. 분석기법', '4. 분석데이터'], 'answer': '2', 'explanation': '분석준비도의 진단 영역으로 분석결과는 올바르지 않은 항목이다.'}, {'question_no': '08', 'question': '다음 중 정형, 반정형, 비정형으로 구분하는 빅데이터 특성으로 옳은 것은?', 'choices': ['1. 가치', '2. 규모', '3. 속도', '4. 다양성'], 'answer': '4', 'explanation': '빅데이터 특성으로는 정형, 반정형, 비정형 등으로 구분되며 다양성이 그중 하나이다.'}, {'question_no': '09', 'question': '다음 중 데이터 전처리의 수행단계로 옳은 것은?', 'choices': ['1. 시스템 구현', '2. 데이터 준비', '3. 데이터 분석', '4. 평가 및 전개'], 'answer': '2', 'explanation': '데이터 전처리의 수행 단계 중 데이터 준비가 올바른 항목이다.'}, {'question_no': '10', 'question': '다음 중 데이터 사이언스에 대한 설명으로 옳은 것은?', 'choices': ['1. 의학, 공학 등 다양한 연구 분야에 적용되고 있다.', '2. 데이터 처리 시점이 사후 처리에서 사전 처리로 이동하였다.', '3. 데이터의 가치 판단 기준이 양보다 질로 그 중요도가 달라졌다.', '4. 단순한 상관 관계 중심에서 이론적 인과 관계로 변화되는 경향이 있다.'], 'answer': '1', 'explanation': '데이터 사이언스는 다양한 연구 분야에 적용되며 의학, 공학 등에서 중요한 역할을 하고 있다.'}]}
    기출_VER1_DECK1.apkg 파일이 생성되었습니다.
    Processing image 2/17
    json.loads
    {'questions': [{'question_no': '11', 'question': '다음 중 데이터 거버넌스의 구성 요소로 옳지 않은 것은?', 'choices': ['1. 원칙', '2. 조직', '3. 프로세스', '4. IT 인프라'], 'answer': '4', 'explanation': '데이터 거버넌스의 구성 요소로 IT 인프라는 포함되지 않습니다.'}, {'question_no': '12', 'question': '다음 중 데이터 산업에 대한 설명으로 옳지 않은 것은?', 'choices': ['1. 데이터를 관리하고 분석하기 위한 소프트웨어 영역이 있다.', '2. 데이터 그 자체를 제공하거나 이를 가공한 정보를 제공한다.', '3. 데이터 산업을 통해 Human to Human 상호 작용이 높아진다.', '4. 데이터 산업은 인프라 영역과 서비스 영역으로 구성되어 있다.'], 'answer': '3', 'explanation': '데이터 산업을 통해 Human to Human 상호 작용이 높아지지 않습니다.'}, {'question_no': '13', 'question': '다음 중 빅 데이터 플랫폼의 계층 구조에 대한 설명으로 옳지 않은 것은?', 'choices': ['1. 최상단에 소프트웨어 계층이 있으며, 아래로 플랫폼 계층, 인프라 스트럭쳐 계층, 하드웨어 계층이 존재한다.', '2. 소프트웨어 계층에서는 빅 데이터 애플리케이션을 구성한다.', '3. 인프라 스트럭쳐 계층에서는 자원 배치와 스토리지 관리를 제공한다.', '4. 플랫폼 계층에서는 빅 데이터 애플리케이션을 실행하기 위한 플랫폼을 제공한다.'], 'answer': '1', 'explanation': '빅 데이터 플랫폼의 계층 구조에서 최상단에 소프트웨어 계층이 위치하는 설명이 옳지 않습니다.'}, {'question_no': '14', 'question': '다음 중 분석 마스터 플랜에 대한 설명으로 옳은 것은?', 'choices': ['1. 데이터 분석 기획의 특성을 고려하지 않습니다.', '2. 분석 과제의 중요도나 난이도는 고려하지 않습니다.', '3. 중장기적 관점의 수행 계획을 수립하는 절차이다.', '4. 그 과제의 목적이나 목표에 따라 부분적인 방향성을 제시한다.'], 'answer': '4', 'explanation': '분석 마스터 플랜은 과제의 목적이나 목표에 따라 부분적인 방향성을 제시합니다.'}, {'question_no': '15', 'question': '다음 중 데이터 분석을 통한 개선 사항을 도출하는 단계로 옳은 것은?', 'choices': ['1. 모델 개발', '2. 분석 목표 수립', '3. 도메인 이슈 도출', '4. 프로젝트 계획 수립'], 'answer': '3', 'explanation': '데이터 분석을 통한 개선 사항을 도출하는 단계에서는 도메인 이슈를 도출합니다.'}]}
    기출_VER1_DECK2.apkg 파일이 생성되었습니다.
    Processing image 3/17
    json.loads
    {'questions': [{'question_no': '16', 'question': '다음 중 데이터 분석 조직에 대한 설명으로 옳지 않은 것은?', 'choices': ['1. 기능형은 특정 현업 부서에 국한된 협소한 분석을 수행할 가능성이 높다.', '2. 집중형은 전사 분석 업무를 별도의 전담 조직에서 수행하므로 중복되지 않는다.', '3. 분산형은 분석 전문 인력을 현업 부서에 배치하여 분석 업무를 신속하게 수행한다.', '4. 조직 구조는 집중형, 기능형, 분산형으로 구분할 수 있으며, 기능형은 DSCoE 조직이 없다.'], 'answer': '1', 'explanation': '기능형은 특정 현업 부서에 국한된 협소한 분석을 수행하는 조직 구조를 의미하며, 이는 전체 조직을 대표하는 DSCoE 조직에서도 가능합니다.'}, {'question_no': '17', 'question': '다음 중 데이터를 추출하여 저장하는 기술로 옳은 것은?', 'choices': ['1. ETL', '2. OLAP', '3. Hadoop', '4. Data Mart'], 'answer': '1', 'explanation': ''}, {'question_no': '18', 'question': '다음 중 탐색적 데이터 분석 (EDA)에 대한 설명으로 옳지 않은 것은?', 'choices': ['1. 데이터 구조를 파악할 수 있다.', '2. 시각화 도구를 이용하여 수행할 수 있다.', '3. 분석 모델을 선정하고 구성하기 위한 절차로 볼 수 있다.', '4. 주성분 분석 (PCA)은 탐색적 데이터 분석에 포함되지 않는다.'], 'answer': '4', 'explanation': '주성분 분석은 탐색적 데이터 분석 과정에서 중요한 한 부분입니다.'}, {'question_no': '19', 'question': '다음 중 분산 파일 시스템에 대한 설명으로 옳지 않은 것은?', 'choices': ['1. 네트워크로 공유하는 여러 호스트의 파일에 접근할 수 있는 파일 시스템이다.', '2. 데이터를 분산하여 저장하면 데이터 추출 및 가공 시 빠르게 처리할 수 있다.', '3. 대표적으로 GFS (Google File System), HDFS (Hadoop Distributed File System)가 있다.', '4. 이기종 데이터 저장 장치를 하나의 데이터 서버에 연결하여 총괄적으로 데이터를 저장 및 관리하는 시스템이다.'], 'answer': '4', 'explanation': ''}, {'question_no': '20', 'question': '다음 중 병렬 DBMS에 대한 설명으로 옳지 않은 것은?', 'choices': ['1. 분산 아키텍처를 가지고 있다.', '2. 데이터 중복의 최소화로 관계형 DBMS보다 성능이 우수하다.', '3. 데이터 파티셔닝과 데이터 병렬 처리를 통해 고성능을 제공한다.', '4. 데이터를 복제하여 분산한 관계로 데이터 변경에 따른 관리 비용이 발생한다.'], 'answer': '2', 'explanation': ''}]}
    기출_VER1_DECK3.apkg 파일이 생성되었습니다.
    Processing image 4/17
    json.loads
    {'questions': [{'question_no': '21', 'question': '다음 아래와 같은 분포 함수를 가지는 확률 분포의 정의로 옳은 것은?', 'choices': ['1. 기하 분포', '2. 포아송 분포', '3. 정규 분포', '4. 이항 확률 분포'], 'answer': '3', 'explanation': '이 문제에서 제시된 확률 분포 함수는 정규 분포를 나타냅니다.'}, {'question_no': '22', 'question': '2, 4, 6, 8, 10의 표본 평균값과 표본 분산을 구하시오.', 'choices': ['1. 평균 6, 분산 8', '2. 평균 6, 분산 10', '3. 평균 5, 분산 8', '4. 평균 6, 분산 7'], 'answer': '4', 'explanation': '주어진 숫자 집합의 평균값은 6이며 표본 분산은 7입니다.'}, {'question_no': '23', 'question': '아래 세 학생의 성적을 최대 최소 정규화하여 모두 합한 값은?', 'choices': ['1. 0.5', '2. 3.5', '3. 2.1', '4. 4.2'], 'answer': '3', 'explanation': '세 학생의 성적을 최대 최소 정규화하여 모두 합하면 2.1이 됩니다.'}, {'question_no': '24', 'question': '다음 중 노이즈를 제거하는 방법이 아닌 것은?', 'choices': ['1. Smoothing', '2. 정규화', '3. 이산화', '4. 이동 평균 (Moving Average)'], 'answer': '3', 'explanation': '이산화는 노이즈를 제거하는 방법 중 하나가 아닙니다.'}, {'question_no': '25', 'question': '독립 변수 12개와 절편을 포함하는 회귀 모델에서, 독립 변수 1개당 범주 3가지를 가지면 회귀 계수는?', 'choices': ['1. 24', '2. 25', '3. 36', '4. 37'], 'answer': '3', 'explanation': '독립 변수가 12개이며 각 변수당 3가지 범주를 가질 때 총 회귀 계수는 36입니다.'}]}
    기출_VER1_DECK4.apkg 파일이 생성되었습니다.
    Processing image 5/17
    json.loads
    {'questions': [{'question_no': '26', 'question': '핫 인코딩 에 대한 설명 중 틀린 것은?', 'choices': ['1. 공간 효율이 좋다.', '2. 범주형 변수를 수치형 변수로 변환하는 방법 중 하나이다.', '3. 범주 간의 거리 계산이 의미가 없을 수 있다.', '4. 각 범주를 명확하게 이진 변수로 표현하기 때문에 해당 범주가 모델의 결과에 어떤 영향을 미치는지 파악할 수 있다.'], 'answer': '3', 'explanation': '<해설 내용>'}, {'question_no': '27', 'question': '비정형 데이터의 특성에 대한 설명 중 맞는 것은?', 'choices': ['1. NoSQL만 사용한다.', '2. 데이터 레이크보다 데이터 웨어하우스를 사용한다.', '3. 다양한 형식과 구조를 가진다.', '4. 전통적인 정형 데이터보다 아직은 그 양이 상대적으로 적다.'], 'answer': '3', 'explanation': '<해설 내용>'}, {'question_no': '28', 'question': '클래스 불균형에 대해 옳지 않은 것은?', 'choices': ['1. Weight Balancing으로 처리가 불가능하다.', '2. 언더 샘플링 혹은 오버 샘플링으로 해결할 수 있다.', '3. 클래스의 개수와 무관하다.', '4. 언더 샘플링과 오버 샘플링은 조합하여 사용이 가능하다.'], 'answer': '1', 'explanation': '<해설 내용>'}, {'question_no': '29', 'question': '파생 변수에 대한 예시와 설명 중 옳지 않은 것은?', 'choices': ['1. 매출에서 총 매출액을 계산한다.', '2. 결측치를 주변 값으로 채운다.', '3. 모델의 설명력을 향상시키며, 예측 능력을 개선하는 데 도움을 줄 수 있다.', '4. 키와 몸무게 변수를 조합하여 체질량 지수(BMI)를 계산한다.'], 'answer': '2', 'explanation': '<해설 내용>'}, {'question_no': '30', 'question': '머신 러닝과 딥 러닝에 대한 설명 중 옳지 않은 것은?', 'choices': ['1. 머신 러닝은 주어진 데이터 패턴을 학습하고 유추하는 것이다.', '2. 인공 지능(AI)의 하위 집합이다.', '3. 머신 러닝은 딥 러닝의 일부이다.', '4. 컴퓨터 성능에 따라 처리 성능이 달라진다.'], 'answer': '4', 'explanation': '<해설 내용>'}]}
    기출_VER1_DECK5.apkg 파일이 생성되었습니다.
    Processing image 6/17
    json.loads
    {'questions': [{'question_no': '31', 'question': '주성분 분석 ( PCA ) 에 대한 설명으로 옳지 않은 것은 ?', 'choices': ['1. 비정 방 행렬 인 음 상관 행렬 의 곱 으로 바꾸어 주성분 분석 의 대상 으로 활용 한다.', '2. 주성분 분석 에서는 데이터 행렬 을 비음 수 행렬 로 가정 하는 경우 도 있다.', '3. 고유 값 이 큰 순서 대로 주성분 을 선택 하여 데이터 의 변동성 을 가장 잘 설명 하는 성분 을 찾는다.', '4. 주성분 분석 은 차원 축소, 데이터 시각화, 변수 선택, 잡음 제거 등 다양한 분야 에서 활용 된다.'], 'answer': '2', 'explanation': '주성분 분석에서는 데이터 행렬을 비정 방 행렬로 가정하는 경우는 없습니다.'}, {'question_no': '32', 'question': '다음과 같이 통계 결과를 시각화한 그림의 정의로 옳은 것은 ?', 'choices': ['1. 박스 플롯', '2. 히스토그램', '3. 산점도', '4. 막대 그래프'], 'answer': '1', 'explanation': '통계 결과를 시각화한 그림을 박스 플롯이라고 합니다.'}, {'question_no': '33', 'question': '다음 중 연속형 변수 가 아닌 것은 ?', 'choices': ['1. 키', '2. 실내 온도', '3. 혈액형', '4. 책 두께'], 'answer': '3', 'explanation': '혈액형은 연속형 변수가 아닌 범주형 변수입니다.'}, {'question_no': '34', 'question': '데이터 이상 값 발생 원인으로 옳지 않은 것은 ?', 'choices': ['1. 측정 오류 (Measurement Error)', '2. 처리 오류 (Processing Error)', '3. 표본 오류 (Sampling Error)', '4. 보고 오류 (Reporting Error)'], 'answer': '4', 'explanation': '보고 오류는 데이터 이상 값 발생 원인에 해당하지 않습니다.'}]}
    기출_VER1_DECK6.apkg 파일이 생성되었습니다.
    Processing image 7/17
    json.loads
    {'questions': [{'question_no': '35', 'question': '35 기초 통계량 에 대해 옳지 않은 설명 은 ?', 'choices': ['1. 사 분위수 는 3 분위 에서 1 분위수 를 뺀 것이다.', '2. 왜 도 는 분포 의 기울어 진 정도 를 설명한 통계량 이다.', '3. 첨도 값 이 3 에 가까우 면 정규 분포 와 비슷 하다.', '4. 변동 계수 는 측정 단위 가 서로 다른 자료 를 비교 하고자 할 때 쓰인다.'], 'answer': '4', 'explanation': '변동 계수는 상대적이고 측정 단위가 서로 다른 자료를 비교하는 데 사용되며, 다른 선택지들은 통계량에 대한 정확한 설명을 제공하고 있습니다.'}, {'question_no': '36', 'question': '다음 보기 중 나머지 와 성질 이 다른 것은 ?', 'choices': ['1. 다항 분포', '2. 포아송 분포', '3. 기하 분포', '4. 지수 분포'], 'answer': '3', 'explanation': '기하 분포는 공비가 일정한 기하수열을 따르는 확률적 분포를 나타내며, 다른 분포들과 성질이 다릅니다.'}, {'question_no': '37', 'question': '다음 중 결측치 를 처리 하는 방법 으로 적절 하지 않은 것은 ?', 'choices': ['1. 다중 대체 법', '2. 단순 대체 법', '3. 완전 삭제 법', '4. 회귀 대체 법'], 'answer': '2', 'explanation': '단순 대체법은 결측치를 대체하는데 사용되지만, 다른 선택지들보다 일반적이고 적절한 방법이 아닙니다.'}, {'question_no': '38', 'question': '이상치 처리 및 평가 에 대한 설명 으로 옳지 않은 것은 ?', 'choices': ['1. 이상 치를 평균값 으로 대체 해도 결 측값 대체 와 같이 신뢰성 이 저하 되지 는 않는다.', '2. Z- 스코어 , 사 분위수 범위 ( IQR ) , 표준 편차 등 의 기준 을 사용 하여 이상 치를 평가 하는 방법 도 있다.', '3. 도메인 전문가 의 지식 과 경험 을 활용 하여 데이터 의 이상 치를 식별 할 수 있다.', '4. 상자 그림 ( Box Plot ) , 히스토그램 , 산점도 등과 같은 기법 을 사용 하여 이상 치를 확인할 수 있다.'], 'answer': '1', 'explanation': '이상치를 평균값으로 대체할 때 신뢰성이 저하될 수 있고, 다른 선택지들은 이상치 처리 및 평가에 대한 올바른 설명을 제공하고 있습니다.'}, {'question_no': '39', 'question': '다음 아래와 같은 시계열 분포도 에 대해서 옳은 것은 ?', 'choices': ['1. 값 A B C 시간', '2. A - B . B - C 로 구간 을 나누면 유의 한 상관 관계 발견 이 가능 하다.', '3. 전 구간 을 2 차 함수 로 근사 하면 제곱항 x 의 부호 는 마이너스 가 될 것이다.', '4. A - B 구간 은 상관 관계 가 음 이고 B - C 구간 은 상관 관계 가 양 이다.'], 'answer': '2', 'explanation': '시계열 분포도에서 A - B 및 B - C 간의 상관 관계를 활용해 유의한 관계를 발견할 수 있습니다.'}]}
    기출_VER1_DECK7.apkg 파일이 생성되었습니다.
    Processing image 8/17
    json.loads
    {'questions': [{'question_no': '40', 'question': '데이터 정제에 대한 설명으로 옳지 않은 것은?', 'choices': ['1. 데이터를 이해하기 쉽게 변환한다.', '2. 처리 데이터가 많은 경우 난수 발생 기법에 의한 임의의 데이터 축소를 실시한다.', '3. 데이터가 다양한 형식으로 저장되어 있는 경우, 일관된 형식으로 표준화 과정이 필요하다.', '4. 이상치를 탐지하고 적절한 처리 방법을 적용하여 제거하거나 보정한다.'], 'answer': '2', 'explanation': '데이터 정제 과정에서 처리 데이터가 많은 경우 난수 발생 기법을 사용하여 임의의 데이터를 축소하는 것은 올바른 방법이 아닐 수 있습니다.'}, {'question_no': '41', 'question': '인공 신경망 학습 모델 중 업데이트 게이트와 리셋 게이트를 사용하여 장기 의존성 문제를 보완한 모델은?', 'choices': ['1. RNN', '2. CNN', '3. GRU', '4. LSTM'], 'answer': '4', 'explanation': '리셋 게이트와 업데이트 게이트를 사용하여 장기 의존성 문제를 보완한 모델은 LSTM(Long Short-Term Memory)이다.'}, {'question_no': '42', 'question': '다음 보기 중 혼동 행렬에 관한 내용으로 옳지 않은 것은?', 'choices': ['1. 재현율은 TP / (TP + FN)이다.', '2. F1 score는 정밀도와 재현율의 기하평균이다.', '3. 정확도는 (TP + TN) / (TP + TN + FP + FN)이다.', '4. 정밀도는 TP / (TP + FP)이다.'], 'answer': '2', 'explanation': 'F1 score는 정밀도와 재현율의 조화평균을 나타내는 지표이며, 정밀도와 재현율의 기하평균이 아닙니다.'}, {'question_no': '43', 'question': '흡연자 200명 중 폐암 환자가 20명이고, 비흡연자 200명 중 폐암 환자가 4명인 경우, 흡연 여부에 대한 폐암 오즈비 값은?', 'choices': ['1. 1', '2. 24.33', '3. 35.44', '4. 46.55'], 'answer': '4', 'explanation': '흡연자와 비흡연자의 폐암 발생 비율을 비교하여 오즈비를 계산하면 결과가 46.55가 나옵니다.'}, {'question_no': '44', 'question': '종속 변수가 범주형이고, 독립 변수가 범주형 변수 하나가 아닌 연속형이거나 둘 이상일 때의 예측 모델은?', 'choices': ['1. 다중 선형 회귀', '2. 다중 로지스틱 회귀', '3. 서포트 벡터 머신', '4. 다층 퍼셉트론'], 'answer': '3', 'explanation': '종속 변수가 범주형이고, 독립 변수가 범주형 변수 하나가 아닌 연속형이거나 둘 이상일 때에는 서포트 벡터 머신이 적합한 예측 모델입니다.'}, {'question_no': '45', 'question': '다음 중 시계열 데이터에서의 공분산 기법을 뜻하는 것은?', 'choices': ['1. 지니 계수', '2. 엔트로피 계수', '3. 자기 상관', '4. 실루엣 계수'], 'answer': '3', 'explanation': '시계열 데이터에서 공분산 기법을 뜻하는 것은 자기상관이며, 시계열 데이터 간 자기상관을 분석하여 패턴을 파악합니다.'}]}
    기출_VER1_DECK8.apkg 파일이 생성되었습니다.
    Processing image 9/17
    json.loads
    {'questions': [{'question_no': '46', 'question': '다중 공선성을 평가하는 지표는?', 'choices': ['1. 분산 팽창 지수 (VIF)', '2. Mallow의 Cp 통계량', '3. 스튜던트 잔차', '4. AIC'], 'answer': '1', 'explanation': '다중 공선성을 평가하기 위한 주요 지표로는 분산 팽창 지수(VIF)가 사용됩니다.'}, {'question_no': '47', 'question': '다음 중 의사 결정 나무의 알고리즘이 아닌 것은?', 'choices': ['1. CART', '2. C45', '3. CHAID', '4. C5.0'], 'answer': '3', 'explanation': 'CHAID는 의사 결정 나무의 알고리즘 중에 해당되지 않습니다.'}, {'question_no': '48', 'question': '다음 중 다중 선형 회귀 평가 지표에 해당하는 것은?', 'choices': ['1. MSE', '2. AIC', '3. BIC', '4. AUC'], 'answer': '3', 'explanation': '다중 선형 회귀 평가 지표 중 BIC(Bayesian Information Criterion)가 해당됩니다.'}, {'question_no': '49', 'question': '랜덤 포레스트 기법에 대한 설명으로 옳지 않은 것은?', 'choices': ['1. 약 분류기를 결합하여 강 분류기를 만드는 기법이다.', '2. 트리로 만든 예측은 다른 트리들과 상관 관계가 작아야 한다.', '3. 부스팅을 사용하여 부트스트랩된 훈련 표본들에 대해 다수의 의사 결정 트리를 만든다.', '4. 알파 컷을 사용한다.'], 'answer': '4', 'explanation': '랜덤 포레스트에서는 알파 컷을 사용하는 것은 아닙니다.'}, {'question_no': '49', 'question': '다음의 의사 결정 나무에서 x 값 구하기', 'choices': ['True', 'False', 'True', 'False', 'x &lt; 12', 'x &gt; 5', 'True', '/', 'True', 'False', 'True', '/', 'X &lt; 7'], 'answer': '3', 'explanation': '주어진 의사 결정 나무에 따라 x 값을 구해보면 x = 9, X2 = 6이 됩니다.'}]}
    기출_VER1_DECK9.apkg 파일이 생성되었습니다.
    Processing image 10/17
    json.loads
    {'questions': [{'question_no': '51', 'question': '다음 보기 중 결정 계수에 대한 설명으로 잘못된 것은?', 'choices': ["1. 독립 변수의 수가 적어 지면 수정된 결정 계수 R '는 커진다.", '2. 결정 계수는 표본수가 증가하면 커지는 경향이 있다.', '3. 결정 계수는 독립 변수 개수가 증가하면 커진다.', '4. 모형에 적합하지 않은 독립 변수가 투입되면 결정 계수가 증가하는 반면 수정된 결정 계수는 감소한다.'], 'answer': '4', 'explanation': '결정 계수는 모형에 적합하지 않은 독립 변수가 투입되면 결정 계수가 증가하는 반면, 수정된 결정 계수는 감소합니다.'}, {'question_no': '52', 'question': '다음 보기 중 시계열 데이터 분석에 관한 것으로 옳지 않은 것을 모두 고른 것은?', 'choices': ['1. 추세 변동은 장기적인 추세 경향이 나타나는 것이다.', '2. 횡단면처럼 종단면은 관측 값 간의 독립성이 중요하다.', '3. 지수 평활법은 과거 값에 높은 가중치를, 최근 값에 작은 가중치를 부여한다.', '4. 이동 평균법은 관측 값 전부에 동일한 가중치를 부여하고 평균을 계산하여 예측한다.'], 'answer': '2', 'explanation': '횡단면처럼 종단면은 관측 값 간의 독립성이 중요하지 않습니다.'}, {'question_no': '53', 'question': 'Causal Analysis 대한 내용으로 옳지 않은 것은?', 'choices': ['1. Causal Inference에서는 어떠한 사건의 원인을 알지만 원인이 되는지 아닌지를 의심이 되는 입력을 따로 정의할 수 있다.', '2. Causal Discovery는 어떤 현상 자체, 즉 Y를 스스로 정의할 수 있는 방법론이다.', '3. Causal Discovery는 데이터 칼럼을 독립 변수 X와 종속 변수 Y로 나누어 정의한다.', '4. 인접 행렬(Adjacency Matrix)을 상호 연결성을 나타내는 지표로 사용된다.'], 'answer': '2', 'explanation': 'Causal Discovery는 어떤 현상 자체, 즉 Y를 스스로 정의할 수 있는 방법론이 아닙니다.'}, {'question_no': '54', 'question': '다중 선형 회귀 모델에서 가정되는 내용이 아닌 것은?', 'choices': ['1. 오차항은 종속 변수와 선형 관계가 있다.', '2. 오차항은 각 독립 변수와 독립적이다.', '3. 각 독립 변수는 종속 변수와 선형 관계에 있다.', '4. 오차항은 평균이 0이고 분산이 일정한 정규 분포를 갖는다.'], 'answer': '3', 'explanation': '다중 선형 회귀 모델에서는 각 독립 변수가 종속 변수와 선형 관계에 있는 것이 아닙니다.'}, {'question_no': '55', 'question': '다음 중 변동 계수에 대한 설명으로 옳은 것은?', 'choices': ['1. 측정 단위가 동일한 자료 간의 흩어진 정도를 상대적으로 비교한다.', '2. 분산을 중심으로 한 산포의 상대적인 척도를 나타내는 수치이다.', '3. 변동 계수가 클수록 상대적으로 분포가 넓어진다.', '4. 값이 작을수록 상대적인 차이가 크다고 할 수 있다.'], 'answer': '3', 'explanation': '변동 계수가 클수록 상대적으로 분포가 넓어집니다.'}]}
    기출_VER1_DECK10.apkg 파일이 생성되었습니다.
    Processing image 11/17
    json.loads
    {'questions': [{'question_no': '56', 'question': '통계적 추론에 대한 설명 중 잘못된 것은?', 'choices': ['1. 모집단을 통해 표본집단을 추론한다.', '2. 통계적 추론의 목적은 추정과 가설 검정에 있다.', '3. 점 추정은 모집단의 특성을 하나의 수치로 추정한다.', '4. 신뢰구간을 추정할 때 모분산을 알고 있다면 표본의 크기와 관계 없이 정규분포를 사용한다.'], 'answer': '4', 'explanation': '통계적 추론에서 신뢰구간을 추정할 때 모분산을 알고 있다면 표본의 크기와 관계 없이 정규분포를 사용할 수 있습니다.'}, {'question_no': '57', 'question': '회귀분석 모형의 구축 절차를 순서대로 나열한 것은?', 'choices': ['1. 독립변수와 종속변수 설정 - 회귀계수 추정 - 독립변수 별 회귀계수 유의성 검정 - 모형 유의성 검정', '2. 회귀계수 추정 - 독립변수와 종속변수 설정 - 독립변수 별 회귀계수 유의성 검정 - 모형 유의성 검정', '3. 독립변수와 종속변수 설정 - 모형 유의성 검정 - 회귀계수 추정 - 독립변수 별 회귀계수 유의성 검정', '4. 독립변수와 종속변수 설정 - 독립변수 별 회귀계수 유의성 검정 - 회귀계수 추정 - 모형 유의성 검정'], 'answer': '4', 'explanation': '회귀분석 모형의 구축 절차는 독립변수와 종속변수 설정, 독립변수 별 회귀계수 유의성 검정, 회귀계수 추정, 그리고 모형 유의성 검정 순서로 진행됩니다.'}, {'question_no': '58', 'question': '인공신경망에서 학습 시에 과적합 방지 방법으로 적절하지 않은 것은?', 'choices': ['1. 입력노드 수를 줄인다.', '2. 가중치 절대값을 최대로 한다.', '3. epoch 수를 줄인다.', '4. hidden layer 수를 줄인다.'], 'answer': '2', 'explanation': '인공신경망에서 가중치 절대값을 최대로 하는 것은 과적합을 방지하는 올바른 방법 중 하나가 아닙니다.'}, {'question_no': '59', 'question': '부스팅에 대한 설명으로 옳지 않은 것은?', 'choices': ['1. 가중치로 약 분류기를 강 분류기로 만든다.', '2. 보팅에 비해 에러가 적다.', '3. 동시병렬적으로 학습한다.', '4. 속도가 상대적으로 느리며 오버피팅될 가능성이 있다.'], 'answer': '3', 'explanation': '부스팅은 동시병렬적으로 학습하는 것이 아닌 순차적으로 강 분류기를 만들어가는 방법입니다.'}]}
    기출_VER1_DECK11.apkg 파일이 생성되었습니다.
    Processing image 12/17
    json.loads
    {'questions': [{'question_no': '60', 'question': '60 아래 이항 로지스틱 회귀 분석 모형 의 회귀 계수 에 대한 설명 으로 옳은 것은 ? ( 단 , B , O )', 'choices': ['1. xj 가 1 단위 증가 하면 오즈 는 ef 배 증가 한다.', '2. xj 가 1 단위 증가 하면 오즈 비 는 배 증가 한다.', "3. x 가 1 단위 증가 하면 오즈 는 e ' 배 증가 한다.", "4. xj 가 1 단위 증가 하면 오즈 비 는 e ' 배 증가 한다."], 'answer': '4', 'explanation': '이항 로지스틱 회귀 모형에서 xj가 1단위 증가할 때 오즈 비는 e^Bj만큼 증가합니다.'}, {'question_no': '61', 'question': '61 SVM 의 하이퍼 파라미터 최적화 과정 에서 두 명의 분석가 의 분석 결과 를 동일 하게 하기 위한 방법 으로 가장 적합한 것은?', 'choices': ['1. Leave - One - Out 교차 검증 하', '2. 5 - fold 교차 검증', '3. Train - Validation - Test Process', '4. 부트 스트래핑'], 'answer': '3', 'explanation': '하이퍼 파라미터 최적화를 위해 Train-Validation-Test Process를 사용하여 두 분석가의 결과를 동일하게 맞출 수 있습니다.'}, {'question_no': '62', 'question': '62 초 매개 변수 튜닝 알고리즘 에 대한 설명 으로 맞지 않은 것은?', 'choices': ['1. 그리드 서치 ( Grid Search ) 는 정해진 범위 내 에서 가능한 모든 조합 을 시도 한다.', '2. 랜덤 서치 ( Random Search ) 는 정해진 범위 내 에서 랜덤 하게 초 매개 변수 를 추출 하여 시도 한다.', '3. 베이지안 최적화 ( Bayesian Optimization ) 는 이전 에 학습 한 결과 를 참고 하여 초 매개 변수 를 설정 한다.', '4. AdaGrad 는 분석가 의 경험 에 따라 값 을 조절 한다.'], 'answer': '4', 'explanation': 'AdaGrad는 경사 하강 알고리즘 중 하나로, 초 매개 변수를 경험에 따라 조절하지 않고 학습률을 조절하는 방법입니다.'}]}
    기출_VER1_DECK12.apkg 파일이 생성되었습니다.
    Processing image 13/17
    json.loads
    {'questions': [{'question_no': '63', 'question': '다음 혼동 행렬을 보고 맞는 것을 고르시오. 실제 답이 True인 것은?', 'choices': ['1. False', '2. True', '3. True False', '4. True Positive'], 'answer': '2', 'explanation': '혼동 행렬 (Confusion Matrix)에서 실제로 Positive인 것을 Positive로 올바르게 판단한 경우를 True Positive라고 하며, 이는 True에 해당하므로 정답은 True입니다.'}, {'question_no': '64', 'question': '군집화 알고리즘 중에서 군집의 수를 지정하지 않아도 되는 것은?', 'choices': ['1. K-Means Clustering', '2. DBSCAN', '3. Gaussian Mixture Model', '4. K-Median Clustering'], 'answer': '2', 'explanation': 'DBSCAN은 군집의 수를 지정하지 않아도 자동으로 결정될 수 있는 밀도 기반 군집화 알고리즘입니다.'}, {'question_no': '65', 'question': '인포그래픽 유형 중 역사적 사건이나 프로젝트 진행 상황 등을 시간 순으로 나열하여 전달하는 데 적합한 것은?', 'choices': ['1. 프로세스 다이어그램', '2. 타임 라인', '3. 지도', '4. 스토리 텔링'], 'answer': '2', 'explanation': '타임 라인은 시간 순으로 사건을 나열하거나 프로젝트 진행 상황을 시각적으로 보여주는 데 적합한 인포그래픽 유형입니다.'}, {'question_no': '66', 'question': '인포그래픽 유형 중 주제, 내용의 연관성을 중요시 여기는 유형은?', 'choices': ['1. 타임 라인', '2. 콘셉트 맵', '3. 스토리 텔링', '4. 비교 분석'], 'answer': '3', 'explanation': '스토리 텔링은 주제와 내용의 연관성을 중요시하여 효과적으로 전달하는 인포그래픽 유형입니다.'}, {'question_no': '67', 'question': '비교 시각화 도구에 대한 설명으로 맞지 않은 것은?', 'choices': ['1. 두 독립된 변수의 분포를 비교해서 보여줄 때 사용된다.', '2. 히트 맵은 값의 분포를 색(온도)으로 표현하여 시각적인 효과를 준다.', '3. 셰르노프 페이스는 데이터 표현에 따라 달라지는 차이를 얼굴의 모양으로 나타낸다.', '4. 스타 차트는 하나의 공간에 각각의 변수를 표현하는 몇 개의 축을 그리고, 축에 표시된 해당 변수의 값들을 별들의 개수로 표현한다.'], 'answer': '3', 'explanation': '셰르노프 페이스는 얼굴의 표정을 이용하여 감정 상태를 나타내는 데 사용되는 시각화 기법이며, 비교 시각화 도구에 대한 설명으로 적합하지 않습니다.'}]}
    기출_VER1_DECK13.apkg 파일이 생성되었습니다.
    Processing image 14/17
    json.loads
    {'questions': [{'question_no': '68', 'question': '시간 시각화에 대한 설명으로 맞지 않는 것은?', 'choices': [{'1.': '막대 그래프는 가로축을 시간축으로 하여 시간 시각화 도구로 사용할 수 있다.'}, {'2.': '점 그래프는 시간 시각화 도구로 사용할 수 없다.'}, {'3.': '선 그래프는 연속적인 데이터를 표현하는 시간 시각화 도구로 사용할 수 있다.'}, {'4.': '점 그래프의 점과 점 사이를 연결함으로써 선 그래프로 변환할 수 있다.'}], 'answer': '2', 'explanation': '점 그래프는 시간 시각화 도구로 사용할 수 없지만, 나머지 그래프는 시간 시각화에 활용될 수 있습니다.'}, {'question_no': '69', 'question': '다음 보기 중 ROC 곡선에 대한 설명으로 옳은 것은?', 'choices': [{'1.': '특이도가 증가할수록 민감도도 증가한다.'}, {'2.': '곡선 아래 면적이 0.5에 가까울수록 성능이 좋다.'}, {'3.': '로지스틱 회귀 분석 모형의 성능을 측정하는 데 사용할 수 있다. 특이도는 음성인 케이스를 양성으로 잘못 예측한 비율이다.'}], 'answer': '3', 'explanation': 'ROC 곡선은 머신러닝 모델의 성능 평가에 활용되며, 그 아래 면적이 클수록 모델의 성능이 좋다고 평가됩니다.'}, {'question_no': '70', 'question': 'Kolmogorov-Smirnov 검정에 대한 설명으로 맞지 않는 것은?', 'choices': [{'1.': '2개의 집단이 동일한 분포를 이루고 있는지를 검증한다.'}, {'2.': '비모수 검정 방식이다.'}, {'3.': '데이터가 정규 분포를 따르는 지를 검증할 때 사용된다.'}, {'4.': '확률 밀도 함수를 사용하여 두 분포의 차이를 측정한다.'}], 'answer': '3', 'explanation': 'Kolmogorov-Smirnov 검정은 정규성 검정이 아닌 두 분포의 동질성을 검정하는 비모수 검정 방법입니다.'}, {'question_no': '71', 'question': '변수 10,000개 중 1,000개를 선별하여 분석 모형을 만드는 경우 가장 적합하지 않은 것은?', 'choices': [{'1.': '임의의 1,000개 변수를 선택하고 학습하는 과정을 100번 반복한다.'}, {'2.': '1,000개의 변수를 선택한 후 학습 데이터와 검증 데이터로 분할하여 평가한다.'}, {'3.': '변수들 사이의 상관 관계를 분석하여 종속 변수와 관련 있는 독립 변수를 선택한다.'}, {'4.': '분석 대상 도메인에 대한 전문 지식을 활용하여 변수를 선택한다.'}], 'answer': '1', 'explanation': '100번 반복하여 임의의 변수를 선택하는 방법은 모델의 일반화를 저하시킬 수 있으므로 가장 적합하지 않습니다.'}, {'question_no': '72', 'question': 'k-fold 교차 검증 학습 과정 중 올바르지 않은 것은?', 'choices': [{'1.': '데이터셋을 k개의 폴드로 나누고, 이 중 하나를 학습 데이터셋으로 선택하고 나머지 k-1개의 폴드를 검증 데이터셋으로 사용한다.'}, {'2.': '학습과 검증을 k번 반복하여 평균 값으로 모델의 성능을 평가한다.'}, {'3.': '반복으로 얻은 성능 지표들을 평균하여 최종 성능 지표를 계산한다.'}], 'answer': '1', 'explanation': 'k-fold 교차 검증에서는 k-1개의 폴드를 학습에 사용하고 1개의 폴드를 검증에 사용하는 방식으로 반복하여 모델을 평가합니다.'}]}



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[2], line 215
        213 # 패키지 이름 지정
        214 apkg_name = deck_name
    --> 215 run_ocr_pipeline(album_name, deck_name, apkg_name)


    Cell In[2], line 111, in run_ocr_pipeline(album_title, deck_title, apkg_title)
        109 ocr_text = convert_ocr_to_text(document)
        110 file_name = apkg_title + str(idx) + '.apkg'
    --> 111 write_to_file(file_name, generate_anki_deck(ocr_text, deck_title))


    Cell In[2], line 189, in generate_anki_deck(ocr_texts, name)
        187 if question_data['choices'] != '':
        188     question = html.escape(question_data['question'])
    --> 189     choices = html.escape(''.join(question_data['choices']))
        190     answer = f"정답: {question_data['answer']} 해설: {html.escape(question_data['explanation'])}"
        191     note = genanki.Note(
        192         model=anki_my_model,
        193         fields=[question_data['question_no'], question, choices, answer]
        194     )


    TypeError: sequence item 0: expected str instance, dict found



```python

```
