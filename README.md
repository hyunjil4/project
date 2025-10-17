# 🚀 3D FEM Solver using JAX-FEM - Complete Beginner's Guide

이 프로젝트는 3D 유한요소법(FEM) 솔버를 JAX-FEM을 사용해서 구현하고, CPU와 GPU(CUDA) 성능을 비교하는 프로그램입니다.

## 📚 **이 가이드는 누구를 위한 것인가요?**

- **CS 초보자**: 프로그래밍을 처음 배우는 분들
- **Python 초보자**: Python을 처음 사용하는 분들  
- **과학계산 초보자**: 수치해석이나 유한요소법을 처음 접하는 분들
- **GPU 가속 초보자**: CUDA나 GPU 가속을 처음 사용하는 분들

## 🎯 **이 프로그램이 하는 일**

1. **3D 구조물 해석**: 3차원 구조물의 변형과 응력을 계산
2. **CPU vs GPU 성능 비교**: CPU와 GPU 중 어느 것이 더 빠른지 측정
3. **시각화**: 결과를 그래프로 보여줌
4. **자동화**: 모든 과정을 자동으로 실행

---

## 🛠️ **1단계: 기본 소프트웨어 설치**

### **1.1 Python 설치 (Anaconda 사용 - 추천)**

**Anaconda란?** Python과 필요한 라이브러리들을 한 번에 설치해주는 프로그램입니다.

#### **Windows 사용자:**

1. **Anaconda 다운로드**
   - 웹사이트: https://www.anaconda.com/download
   - "Download" 버튼 클릭
   - Python 3.11 버전 선택 (64-bit)

2. **설치**
   - 다운로드한 파일 실행
   - "Just Me" 선택 (개인 사용)
   - "Add Anaconda to PATH" 체크박스 **반드시 체크**
   - "Install" 클릭

3. **설치 확인**
   - `Win + R` 키 누르기
   - `cmd` 입력하고 Enter
   - 명령창에 `python --version` 입력
   - Python 3.11.x 버전이 나오면 성공!

#### **Mac 사용자:**

1. **Anaconda 다운로드**
   - 웹사이트: https://www.anaconda.com/download
   - "Download" 버튼 클릭
   - Python 3.11 버전 선택 (MacOS)

2. **설치**
   - 다운로드한 `.pkg` 파일 실행
   - 설치 과정을 따라 진행
   - "Add Anaconda to PATH" 옵션 선택

3. **설치 확인**
   - 터미널 열기 (Spotlight에서 "터미널" 검색)
   - `python3 --version` 입력
   - Python 3.11.x 버전이 나오면 성공!

#### **Linux 사용자:**

```bash
# 터미널에서 실행
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh
# 설치 과정에서 "yes" 입력하고 Enter 키 누르기
source ~/.bashrc
python3 --version
```

### **1.2 Git 설치 (코드 다운로드용)**

#### **Windows 사용자:**
1. https://git-scm.com/download/win 방문
2. "Download for Windows" 클릭
3. 설치 파일 실행하고 기본 설정으로 설치

#### **Mac 사용자:**
```bash
# 터미널에서 실행
brew install git
# 또는 Xcode Command Line Tools 설치
xcode-select --install
```

#### **Linux 사용자:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install git

# CentOS/RHEL
sudo yum install git
```

---

## 🎮 **2단계: GPU 가속 설정 (선택사항)**

**GPU 가속이란?** 그래픽카드를 사용해서 계산을 더 빠르게 하는 기술입니다.

### **2.1 NVIDIA GPU 확인**

#### **Windows 사용자:**
1. `Win + R` 키 누르기
2. `dxdiag` 입력하고 Enter
3. "Display" 탭에서 "Name" 확인
4. NVIDIA로 시작하면 GPU 가속 가능!

#### **Mac 사용자:**
```bash
# 터미널에서 실행
system_profiler SPDisplaysDataType | grep -i nvidia
```

#### **Linux 사용자:**
```bash
# 터미널에서 실행
lspci | grep -i nvidia
```

### **2.2 CUDA 설치 (NVIDIA GPU가 있는 경우만)**

**CUDA란?** NVIDIA GPU를 사용해서 계산을 가속화하는 도구입니다.

#### **Windows 사용자:**

1. **NVIDIA 드라이버 확인**
   - `Win + X` 키 누르기
   - "장치 관리자" 선택
   - "디스플레이 어댑터" 확장
   - NVIDIA 그래픽카드 확인

2. **CUDA Toolkit 다운로드**
   - 웹사이트: https://developer.nvidia.com/cuda-downloads
   - "Windows" 선택
   - "x86_64" 선택
   - "exe (network)" 다운로드

3. **CUDA 설치**
   - 다운로드한 파일 실행
   - "Express" 설치 선택
   - 설치 완료 후 재부팅

4. **설치 확인**
   - `Win + R` 키 누르기
   - `cmd` 입력하고 Enter
   - `nvcc --version` 입력
   - CUDA 버전이 나오면 성공!

#### **Mac 사용자:**
```bash
# 터미널에서 실행
# Mac은 CUDA를 직접 지원하지 않으므로 CPU만 사용
echo "Mac에서는 CPU만 사용 가능합니다"
```

#### **Linux 사용자:**
```bash
# 터미널에서 실행
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2004-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

---

## 📥 **3단계: 프로젝트 다운로드**

### **3.1 프로젝트 다운로드**

#### **방법 1: Git 사용 (추천)**
```bash
# 터미널/명령창에서 실행
git clone https://github.com/your-username/MHIT36-main.git
cd MHIT36-main/살려줘
```

#### **방법 2: ZIP 파일 다운로드**
1. GitHub 페이지에서 "Code" → "Download ZIP" 클릭
2. ZIP 파일 압축 해제
3. `MHIT36-main/살려줘` 폴더로 이동

### **3.2 파일 확인**
다음 파일들이 있는지 확인하세요:
- `run_experiment.sh` (Linux/Mac용)
- `run_experiment.bat` (Windows용)
- `working_demo.py`
- `requirements.txt`

---

## 🔧 **4단계: 환경 설정**

### **4.1 가상환경 생성**

**가상환경이란?** 프로젝트별로 독립적인 Python 환경을 만드는 것입니다.

#### **Windows 사용자:**
```cmd
# 명령창(cmd)에서 실행
cd MHIT36-main\살려줘
python -m venv jax_fem_env
jax_fem_env\Scripts\activate
```

#### **Mac/Linux 사용자:**
```bash
# 터미널에서 실행
cd MHIT36-main/살려줘
python3 -m venv jax_fem_env
source jax_fem_env/bin/activate
```

**성공하면 명령창 앞에 `(jax_fem_env)`가 나타납니다!**

### **4.2 필요한 라이브러리 설치**

#### **자동 설치 (추천):**

**Windows 사용자:**
```cmd
# 명령창에서 실행 (가상환경 활성화된 상태)
install_dependencies.bat
```

**Mac/Linux 사용자:**
```bash
# 터미널에서 실행 (가상환경 활성화된 상태)
chmod +x install_dependencies.sh
./install_dependencies.sh
```

#### **수동 설치:**
```bash
# 가상환경 활성화된 상태에서 실행
pip install -r requirements.txt
```

---

## 🚀 **5단계: 프로그램 실행**

### **5.1 설치 확인**
```bash
# 가상환경 활성화된 상태에서 실행
python test_installation.py
```

**성공하면:** "✅ All packages installed correctly!" 메시지가 나타납니다.

### **5.2 GPU 확인 (선택사항)**
```bash
# 가상환경 활성화된 상태에서 실행
python check_cuda_version.py
```

**GPU가 있으면:** CUDA 버전과 GPU 정보가 나타납니다.
**GPU가 없으면:** "No GPU detected" 메시지가 나타납니다 (정상입니다).

### **5.3 기본 테스트**
```bash
# 가상환경 활성화된 상태에서 실행
python working_demo.py
```

**성공하면:** CPU와 GPU 성능 비교 결과가 나타납니다.

### **5.4 완전한 실험 실행**

#### **Windows 사용자:**
```cmd
# 명령창에서 실행 (가상환경 활성화된 상태)
run_experiment.bat
```

#### **Mac/Linux 사용자:**
```bash
# 터미널에서 실행 (가상환경 활성화된 상태)
chmod +x run_experiment.sh
./run_experiment.sh
```

---

## 📊 **6단계: 결과 확인**

### **6.1 실행 결과**
프로그램이 성공적으로 실행되면:

1. **화면에 출력되는 내용:**
   ```
   3D FEM SOLVER BENCHMARK: CPU vs GPU (CUDA)
   =============================================
   
   Mesh: 10×10×10 elements
   Nodes: 1,331
   Elements: 1,000
   DOFs: 3,993
   
   Assembly Time:
     CPU:  0.1234 seconds
     GPU:  0.0456 seconds
     Speedup: 2.71x
   
   Solve Time:
     CPU:  0.0567 seconds
     GPU:  0.0234 seconds
     Speedup: 2.42x
   
   Total Time:
     CPU:  0.1801 seconds
     GPU:  0.0690 seconds
     Speedup: 2.61x
   ```

2. **생성되는 파일들:**
   - `performance_comparison.png` - 성능 비교 그래프
   - `results_YYYYMMDD_HHMMSS/` 폴더 - 상세 결과

### **6.2 결과 해석**

- **Speedup > 1**: GPU가 CPU보다 빠름
- **Speedup = 1**: GPU와 CPU 성능이 비슷함
- **Speedup < 1**: CPU가 GPU보다 빠름 (드물게 발생)

---

## 🆘 **문제 해결**

### **문제 1: "python이 인식되지 않습니다"**
**해결방법:**
1. Anaconda 설치 시 "Add to PATH" 옵션을 체크했는지 확인
2. 컴퓨터 재부팅
3. Anaconda Prompt 사용

### **문제 2: "pip이 인식되지 않습니다"**
**해결방법:**
```bash
# 가상환경 활성화 후 실행
python -m pip install --upgrade pip
```

### **문제 3: "NumPy 2.x 오류"**
**해결방법:**
```bash
# 가상환경 활성화 후 실행
python fix_numpy_compatibility.py
```

### **문제 4: "JAX/NumPy 호환성 오류"**
**해결방법:**
```bash
# 가상환경 활성화 후 실행
python fix_jax_numpy_compatibility.py
```

### **문제 5: "CUDA를 찾을 수 없습니다"**
**해결방법:**
1. NVIDIA 드라이버 최신 버전 설치
2. CUDA Toolkit 재설치
3. 컴퓨터 재부팅

---

## 📁 **파일 설명**

### **실행 파일들**
- `run_experiment.sh` - **완전한 실험** (Linux/Mac용)
- `run_experiment.bat` - **완전한 실험** (Windows용)
- `working_demo.py` - **기본 테스트** (추천)
- `run_benchmark.py` - 성능 벤치마크

### **설치 파일들**
- `install_dependencies.sh` - 자동 설치 (Linux/Mac)
- `install_dependencies.bat` - 자동 설치 (Windows)
- `requirements.txt` - 필요한 라이브러리 목록

### **문제 해결 파일들**
- `fix_numpy_compatibility.py` - NumPy 문제 해결
- `fix_jax_numpy_compatibility.py` - JAX/NumPy 문제 해결
- `test_installation.py` - 설치 확인

### **문서 파일들**
- `README.md` - 이 파일 (완전한 가이드)
- `QUICK_START.md` - 빠른 시작 가이드
- `EXECUTION_GUIDE.md` - 실행 순서 가이드
- `CUDA_INSTALLATION_GUIDE.md` - CUDA 설치 가이드

---

## 🎯 **빠른 시작 (한 줄 명령어)**

### **Windows 사용자:**
```cmd
# 1. 프로젝트 다운로드 후
cd MHIT36-main\살려줘
python -m venv jax_fem_env
jax_fem_env\Scripts\activate
install_dependencies.bat
run_experiment.bat
```

### **Mac/Linux 사용자:**
```bash
# 1. 프로젝트 다운로드 후
cd MHIT36-main/살려줘
python3 -m venv jax_fem_env
source jax_fem_env/bin/activate
chmod +x *.sh
./install_dependencies.sh
./run_experiment.sh
```

---

## 📞 **도움이 필요하신가요?**

1. **먼저 시도해보세요:**
   - `python test_installation.py` - 설치 확인
   - `python working_demo.py` - 기본 테스트

2. **문제가 있으면:**
   - `python fix_numpy_compatibility.py` - NumPy 문제 해결
   - `python fix_jax_numpy_compatibility.py` - JAX 문제 해결

3. **여전히 문제가 있으면:**
   - `results_*/` 폴더의 로그 파일 확인
   - 오류 메시지를 정확히 복사해서 문의

---

## 🎉 **축하합니다!**

이제 3D 유한요소법 솔버를 사용할 수 있습니다! 

- **CPU vs GPU 성능 비교**를 통해 하드웨어 성능을 확인할 수 있습니다
- **시각화**를 통해 결과를 쉽게 이해할 수 있습니다
- **자동화**를 통해 복잡한 설정 없이 바로 사용할 수 있습니다

**다음 단계:** `jax_fem_3d_solver.py` 파일을 수정해서 자신만의 문제를 해결해보세요!