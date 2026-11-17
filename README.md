# Numerical Analysis Project



### 프로젝트 다운로드 방법

1. GitHub 저장소 우측 상단의 초록색 **`<> Code`** 버튼을 클릭합니다.
2. **`Download ZIP`** 을 선택해 압축 파일을 다운로드합니다.
3. 원하는 위치에 압축을 해제한 뒤, 해당 폴더에서 터미널(CMD 또는 PowerShell)을 엽니다.


### 실행 방법 1: `uv` 사용 시 (권장)

`uv`가 설치된 환경에서는 별도의 가상환경 생성이나 패키지 수동 설치 없이, 프로젝트 폴더 경로에서 아래 명령어를 입력하면 동일한 환경이 자동으로 구축되며 코드가 실행됩니다.

```
# 1. 가상환경 및 의존성 패키지 사전 동기화
uv sync

# 2. 파이썬 스크립트 실행
uv run 파일명.py

```

### 실행 방법 2: 일반 Python (`pip`) 사용 시

`uv`가 설치되어 있지 않은 환경에서는 표준 `venv`와 `pip`를 이용해 `requirements.txt`에 명시된 패키지를 설치한 후 실행합니다.

#### Windows (CMD / PowerShell)

```
:: 1. 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate

:: 2. 필수 패키지 설치
pip install -r requirements.txt

:: 3. 파이썬 스크립트 실행
python 파일명.py

```

#### macOS / Linux

```
# 1. 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 2. 필수 패키지 설치
pip install -r requirements.txt

# 3. 파이썬 스크립트 실행
python3 파일명.py

```