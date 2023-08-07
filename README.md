# 환경설정 방법

루트 폴더 내에 존재하는 requirements.txt를 사용하여 환경을 설정.  
이 때, pytorch가 무사히 GPU 버전으로 설치되었는지 확인해야 함

```bash
$ pip install -r requirements.txt #가상환경에서 설치하는 것을 권장
```

<br>
</br>

# 데이터 폴더 경로

루트 폴더 내에 해당 링크를 통해 설치한 data.zip, logs.zip을 해제

https://drive.google.com/drive/folders/1cgCAXl_N3WRuX-kNwXcJtfE236Efcsxy?usp=sharing

```bash
BIVI
 ├── data
 │   └── signal
 │           ├── 0
 │           └── 7
 ├── logs
 │   ├── class_0
 │   └── class_7
 ├── Vibration_Train.py
 │ ...
```

<br>
</br>

# 학습 방법

루트 폴더에 존재하는 Vibration_Train.py를 shell에서 실행

```bash
$ python Vibration_Train.py
```

<br>
</br>

# 검증 방법

루트 폴더에 존재하는 Result_analysis.ipynb 파일을 이용하여 real 데이터와 synthetic 데이터에 대한 시각화 분석을 확인할 수 있음

```bash
BIVI
 ├── data
 │   └── signal
 │           ├── 0
 │           └── 7
 ├── logs
 │   ├── class_0
 │   └── class_7
 ├── Vibration_Train.py
 ├── Result_analysis.ipynb
 │ ...
```
