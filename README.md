# smart_city_2025

스마트시티 수업에 대한 내용

## notion 페이지

- wsl : [링크](https://www.notion.so/freshmea/WSL-windows-subsystem-for-linux-232123060ee780e79964ec56e36b5c18?source=copy_link)

## WSL 설치

```shellshell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

wsl --install
wsl --install -d Ubuntu-22.04
wsl --list --verbose
wsl --set-default-version 2
```

## usbipd

```shell

winget install --interactive --exact dorssel.usbipd-win
usbipd list
usbipd bind --busid 4-2
usbipd attach --wsl --busid 7-3

```

# 3일 간의 수업 내용 정리

## 1일차

- 개발 환경 설정 : wsl, colab, vscode, github, opencv 설치
- 딥러닝 모델 개요
- 딥러닝 학습 원리 및 추론 과정
- 딥러닝 모델 최적화 기법
- DNN , CNN 기초 이론
- 강아지와 고양이 사진 CNN 을 이용한 분류 실습
- 딥러닝 및 컴퓨터 비전 개요
- opencv 기본 함수 사용법
  - imshow, imread, VideoCapture, rectangle, putText 등

## 2일차

- YOLO 모델 구조 및 동작 원리
- YOLO seg 모델을 활용한 객체 검출 및 분할 실습
- YOLO pose 모델을 활용한 자세 추정 실습
- YOLO obb 모델을 활용한 회전된 객체 검출 실습
- YOLO cls 모델을 활용한 이미지 분류 실습
- OpenCV 딥러닝 모듈 활용법 ( opencv zoo)
- yunet 모델을 활용한 얼굴 검출 실습
- SFACE 기반 얼굴 인식 실습
- Mediapipe 기반 핸드 포즈 추정 실습
- LPD-YuNet 기반 차량 번호판 검출 실습
- VITTrack 기반 객체 추적 실습
- YOLOTrack 기반 객체 검출 및 추적 실습
