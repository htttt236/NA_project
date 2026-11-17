Numerical Analysis Project

이 프로젝트는 uv를 기반으로 파이썬 환경 및 의존성(numpy, scipy, matplotlib)이 구성되어 있습니다. 아래 방법 중 하나를 선택하여 코드를 실행할 수 있습니다.

방법 1: uv 사용 시 (권장)

uv가 설치된 환경에서는 별도의 가상환경 생성이나 패키지 수동 설치 없이, 프로젝트 폴더 경로에서 아래 명령어를 입력하면 동일한 환경이 자동으로 구축되며 코드가 실행됩니다.

# 1. 가상환경 및 의존성 패키지 사전 동기화
uv sync

# 2. 파이썬 스크립트 실행
uv run 파일명.py


방법 2: 일반 Python (pip) 사용 시

uv가 설치되어 있지 않은 환경에서는 표준 venv와 pip를 이용해 requirements.txt에 명시된 패키지를 설치한 후 실행합니다.

Windows (CMD / PowerShell)

:: 1. 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate

:: 2. 필수 패키지 설치
pip install -r requirements.txt

:: 3. 파이썬 스크립트 실행
python 파일명.py


macOS / Linux

# 1. 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 2. 필수 패키지 설치
pip install -r requirements.txt

# 3. 파이썬 스크립트 실행
python3 파일명.py
