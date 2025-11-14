import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from ultralytics import YOLO

# (선택) 설치되어 있으면 True
try:
    import supervision as sv
    HAS_SV = True
except Exception:
    HAS_SV = False


# 사용할 트래커 설정 파일 매핑
TRACKER_MAP = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}

# =========================
#  ROI / 침입 상태 전역 변수
# =========================
# - roi_points     : 사용자가 화면에 찍은 ROI 꼭짓점 리스트
# - roi_finalized  : ROI 다각형이 확정되었는지 여부
# - id_inside_state: 각 트랙 ID가 현재 ROI 안에 있는지 상태 기록 (밖→안 전환 감지용)
roi_points = []        # [(x, y), ...]
roi_finalized = False  # True가 되면 더 이상 점 추가/수정 X
id_inside_state = {}   # {track_id: bool}


# =========================
#  설정/유틸 함수들
# =========================
def load_tracker_config(tracker_path: str):
    """
    트래커 설정 파일(YAML)을 로드하고,
    주요 설정을 로그에 남긴다.
    """
    try:
        with open(tracker_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        logging.info(f"=== 트래커 설정 로드됨: {tracker_path} ===")
        logging.info(f"트래커 타입: {config.get('tracker_type', 'Unknown')}")

        if config.get("tracker_type") == "botsort":
            logging.info(f"ReID 활성화: {config.get('with_reid', False)}")
            if config.get("with_reid", False):
                logging.info(f"ReID 모델: {config.get('model', 'auto')}")
            logging.info(f"트랙 버퍼: {config.get('track_buffer', 30)} 프레임")
            logging.info(f"매칭 임계값: {config.get('match_thresh', 0.8)}")
            logging.info(f"외형 임계값: {config.get('appearance_thresh', 0.25)}")

        logging.info(f"ReID 사용: {config.get('with_reid', False)}")
        logging.info(f"ReID 모델: {config.get('model', 'auto')}")

        return config

    except Exception as e:
        logging.error(f"트래커 설정 파일 로드 실패: {e}")
        return None


def parse_args():
    """
    커맨드라인 인자를 정의하고 파싱한다.
    """
    p = argparse.ArgumentParser(
        description="YOLO + (ByteTrack/BoT-SORT) MOT + 다각형 ROI 침입 감지 데모"
    )

    # 입력 소스: 0(웹캠) 또는 비디오 파일 경로
    p.add_argument(
        "--source",
        type=str,
        default="0",
        help="영상 경로 혹은 카메라 인덱스 (예: 0)",
    )

    # YOLO 모델 가중치
    p.add_argument(
        "--model",
        type=str,
        default="yolov8l.pt",
        help="Ultralytics YOLO 가중치 (예: yolov8n.pt, yolov8l.pt, yolov8x.pt 등)",
    )

    # 트래커 종류
    p.add_argument(
        "--tracker",
        type=str,
        default="bytetrack",
        choices=list(TRACKER_MAP.keys()),
        help="트래커 선택: bytetrack | botsort",
    )

    # confidence / IoU threshold
    p.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="추론 confidence threshold (낮게 설정하면 더 많이 탐지)",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold (높게 설정하면 박스가 더 합쳐짐)",
    )

    # p.add_argument("--device", type=str, default="0",
    #                help="장치 선택 (예: '0' 또는 'cpu'). 기본은 자동")

    # 결과 영상 저장 경로 (원하면 mp4로 저장)
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="저장할 출력 영상 경로 (예: output.mp4). 지정 안 하면 저장 안 함",
    )

    # 창에 표시 여부
    p.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="윈도우에 실시간 표시 여부",
    )

    # 특정 클래스만 추적하고 싶을 때 (COCO id 사용)
    p.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="특정 클래스만 추적 (COCO id 리스트, 예: --classes 0 2 7)",
    )

    return p.parse_args()


def open_writer(example_frame, save_path: str, fps: int = 30):
    """
    첫 프레임의 크기를 기준으로 비디오 writer를 생성한다.
    """
    h, w = example_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))


# =========================
#  ROI 관련 함수들
# =========================
def mouse_callback(event, x, y, flags, param):
    """
    OpenCV 마우스 콜백.

    - 왼쪽 클릭(LBUTTONDOWN): ROI 꼭짓점 추가
    - 오른쪽 클릭(RBUTTONDOWN): ROI 다각형 확정 (3개 이상일 때만)
    """
    global roi_points, roi_finalized

    # 이미 확정된 상태면 더 이상 점을 수정하지 않는다.
    if roi_finalized:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        # 점 추가
        roi_points.append((x, y))
        print(f"ROI 점 추가: ({x}, {y}) | 총 {len(roi_points)}개")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # 다각형 확정 (최소 3개의 점 필요)
        if len(roi_points) >= 3:
            roi_finalized = True
            print(f"ROI 확정 완료: {len(roi_points)}개 점으로 구성된 다각형")
        else:
            print("ROI 확정을 위해서는 최소 3개의 점이 필요합니다.")


def is_inside_roi(point, roi_pts):
    """
    point = (x, y)가 roi_pts(다각형 꼭짓점 리스트) 안에 있는지 여부를 반환.

    - OpenCV pointPolygonTest 사용
      > 0 : 내부
      = 0 : 경계
      < 0 : 외부
    - 여기서는 '경계 포함' 내부로 처리하기 위해 res >= 0 을 사용.
    """
    if not roi_pts or len(roi_pts) < 3:
        return False

    contour = np.array(roi_pts, dtype=np.int32)
    res = cv2.pointPolygonTest(contour, point, False)
    return res >= 0  # 경계 포함


# =========================
#  메인 처리 루프
# =========================
def main():
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # source가 숫자 문자열이면 웹캠 인덱스로 변환 (예: "0" -> 0)
    source = args.source
    if source.isdigit():
        source = int(source)

    # YOLO 모델 로드
    model = YOLO(args.model)
    logging.info(f"YOLO 모델 로드 완료: {args.model}")

    # 트래커 설정 로드
    tracker_cfg = TRACKER_MAP[args.tracker]
    config = load_tracker_config(tracker_cfg)
    if config is None:
        logging.warning("트래커 설정을 불러오지 못했습니다. 기본 설정으로 진행합니다.")
        
    # ========= ROI를 먼저 설정하고 시작 =========
    if args.show:
        setup_roi_on_first_frame(args.source)
    # ===========================================

    logging.info(f"트래킹 시작 - Source: {source}, Tracker: {args.tracker}")

    # YOLO의 track() 함수 호출 (stream=True로 제너레이터 사용)
    try:
        gen = model.track(
            source=source,
            stream=True,          # 한 프레임씩 yield
            tracker=tracker_cfg,  # ByteTrack 또는 BoT-SORT 설정 파일
            persist=True,         # 객체 ID 유지
            conf=args.conf,
            iou=args.iou,
            # device=args.device,
            classes=args.classes,
        )
        logging.info("트래커 초기화 성공")
    except Exception as e:
        logging.error(f"트래커 초기화 실패: {e}")
        return

    writer = None
    last_time = time.time()
    fps = 0.0
    
    # 누적 침입 횟수 & 마지막 프레임 저장용
    intrusion_count = 0
    last_frame = None

    # 스크린샷 저장 폴더 생성 (프로젝트 폴더 안에 intrusion_shots/)
    screenshot_dir = Path("intrusion_shots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    
    # 누적 침입 횟수 & 마지막 프레임 저장용
    intrusion_count = 0
    last_frame = None

    # supervision이 설치되어 있어도, ROI 색 구분을 위해
    # 여기서는 OpenCV 직접 그리기를 기본으로 사용.
    if HAS_SV:
        logging.info(
            "supervision 패키지가 감지되었지만, "
            "ROI 색상 제어를 위해 OpenCV 직접 그리기를 사용합니다."
        )

    # ROI를 그릴 윈도우 이름
    window_name = "YOLO MOT + ROI Intrusion Demo"
    if args.show:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

    # ===== 메인 프레임 처리 루프 =====
    for i, result in enumerate(gen):
        frame = result.orig_img
        if frame is None:
            continue
        
        # 마지막 프레임을 기억해둔다 (영상 끝난 후 보여주기용)
        last_frame = frame.copy()

        # FPS 계산 (지수 이동 평균을 사용해 조금 부드럽게)
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
        last_time = now

        # 이 프레임에서 ROI 안에 있는 객체 수
        inside_count = 0

        # -----------------------
        #  ROI 화면에 그리기
        # -----------------------
        if args.show:
            # 찍힌 ROI 점이 있다면 점 + 선/다각형을 그려준다.
            if roi_points:
                # 각 점을 작은 원으로 표시
                for p in roi_points:
                    cv2.circle(frame, p, 4, (255, 0, 0), -1)

                pts = np.array(roi_points, np.int32)
                if len(pts) >= 2:
                    # roi_finalized=True 이면 닫힌 다각형, False면 열린 폴리라인
                    cv2.polylines(frame, [pts], roi_finalized, (255, 0, 0), 2)

            # ROI 사용법 안내 문구
            if not roi_finalized:
                msg = "L-Click: add ROI point  |  R-Click: finalize ROI"
            else:
                msg = "ROI fixed"
            cv2.putText(
                frame,
                msg,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 255),
                2,
            )

        # -----------------------
        #  YOLO 추적 결과 처리
        # -----------------------
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            # 첫 프레임에 탐지된 객체 개수 로깅
            if i == 0:
                logging.info(f"첫 번째 프레임에서 {len(boxes)}개 객체 탐지")

            # tensor -> numpy 로 변환
            xyxy = boxes.xyxy.cpu().numpy()
            confs = (
                boxes.conf.cpu().numpy()
                if boxes.conf is not None
                else np.zeros(len(xyxy))
            )
            clss = (
                boxes.cls.cpu().numpy().astype(int)
                if boxes.cls is not None
                else np.zeros(len(xyxy), dtype=int)
            )
            ids = (
                boxes.id.cpu().numpy().astype(int)
                if boxes.id is not None
                else np.array([-1] * len(xyxy), dtype=int)
            )

            # 30프레임마다 현재 활성 트랙 ID 로그
            if i % 30 == 0 and len(ids) > 0:
                active_ids = [tid for tid in ids if tid != -1]
                logging.info(f"활성 트래킹 ID: {active_ids} (총 {len(active_ids)}개)")

            # ===== 박스 그리기 + ROI 침입 여부 체크 =====
            for (x1, y1, x2, y2), cid, conf, tid in zip(xyxy, clss, confs, ids):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                # 바운딩 박스 중심점 계산
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # ROI가 확정된 상태라면, 중심점이 ROI 내부인지 검사
                inside = roi_finalized and is_inside_roi((cx, cy), roi_points)

                # ROI 안: 빨간색, 밖: 초록색
                color = (0, 255, 0)
                if inside:
                    color = (0, 0, 255)
                    inside_count += 1   # ⭐ ROI 내부 객체 개수 카운트

                # 바운딩 박스 & 중심점 표시
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 3, color, -1)

                # 클래스 이름 가져오기 (없으면 id 그대로)
                class_name = model.model.names.get(cid, str(cid))

                # 라벨 텍스트 구성 (ROI 안일 경우 [IN] 표시)
                label = f"ID {tid} | {class_name} {conf:.2f}"
                if inside:
                    label += " [IN]"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                # ----- 침입 이벤트 로그 -----
                if tid is not None and tid != -1:
                    prev_inside = id_inside_state.get(tid, False)
                    if inside and not prev_inside:
                        intrusion_count += 1    # 누적 침입 횟수 증가
                        
                        # 스크린샷 파일 이름 만들기
                        screenshot_path = screenshot_dir / f"intrusion_{intrusion_count:04d}.jpg"

                        # 현재 프레임을 저장 (박스/텍스트가 그려진 상태로 저장하고 싶으면 frame 사용)
                        cv2.imwrite(str(screenshot_path), frame.copy())
                        
                        logging.info(
                            f"⚠️ ROI 침입 감지 - ID {tid}, 클래스 {class_name}, "
                            f"중심 ({cx}, {cy}) | 저장: {screenshot_path}"
                        )
                        
                    id_inside_state[tid] = inside

        # -----------------------
        #  상단 정보 표시 (FPS + IN-ROI 카운트)
        # -----------------------
        # 좌상단 FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Model: {args.model} | Tracker: {args.tracker}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (50, 230, 50),
            2,
        )

        # 우상단 IN-ROI 카운트
        h, w = frame.shape[:2]
        count_text = f"IN-ROI COUNT: {intrusion_count}"
        (tw, th), _ = cv2.getTextSize(count_text,
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.7,
                                      2)
        # 오른쪽에서 10px 안쪽, 위에서 30px 정도
        cv2.putText(
            frame,
            count_text,
            (w - tw - 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),   # 빨간색 글씨
            2,
        )

        # 저장 옵션이 있다면, 첫 프레임에서 writer 생성
        if args.save and writer is None:
            guessed_fps = 30  # 입력 FPS를 모르면 대략 30으로 가정
            writer = open_writer(frame, args.save, fps=guessed_fps)

        # -----------------------
        #  프레임 표시 / 저장
        # -----------------------
        if args.show:
            cv2.imshow(window_name, frame)
            # ESC 키 누르면 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if writer is not None:
            writer.write(frame)

    # 루프 종료 후 리소스 정리
    if writer is not None:
        writer.release()
        
    # 영상이 끝난 뒤에도 마지막 화면을 보여주고,
    # 사용자가 키(ESC 또는 q)를 눌러야 창이 닫히도록 처리
    if args.show and last_frame is not None:
        logging.info("영상이 종료되었습니다. ESC 또는 q 를 누르면 창이 닫힙니다.")
        while True:
            cv2.imshow(window_name, last_frame)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord('q')):   # ESC 또는 q
                break

    if args.show:
        cv2.destroyAllWindows()

def setup_roi_on_first_frame(source):
    """
    영상/웹캠에서 첫 프레임을 읽어서,
    그 위에서 미리 ROI를 설정하는 함수.

    - 좌클릭: 점 추가
    - 우클릭: ROI 확정
    - Enter(또는 Space, 's'): ROI 확정 후 시작
    - ESC: ROI 초기화 후 종료 (ROI 없이 진행)
    """
    global roi_points, roi_finalized

    # 숫자/문자 구분 (main과 동일하게)
    cap_source = source
    if isinstance(cap_source, str) and cap_source.isdigit():
        cap_source = int(cap_source)

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        logging.error(f"ROI 설정용 캡처 오픈 실패: {cap_source}")
        return

    ret, frame = cap.read()
    if not ret or frame is None:
        logging.error("ROI 설정용 첫 프레임을 읽지 못했습니다.")
        cap.release()
        return

    # ROI 상태 초기화
    roi_points = []
    roi_finalized = False

    win_name = "ROI Setup"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_callback)

    logging.info("ROI 설정 모드 진입: 좌클릭으로 점 추가, 우클릭으로 다각형 확정, Enter/Space/'s'로 시작")

    while True:
        disp = frame.copy()

        # 점/폴리라인 그리기 (메인 루프에서 하던 코드와 거의 동일)
        if roi_points:
            for p in roi_points:
                cv2.circle(disp, p, 4, (255, 0, 0), -1)

            pts = np.array(roi_points, np.int32)
            if len(pts) >= 2:
                cv2.polylines(disp, [pts], roi_finalized, (255, 0, 0), 2)

        # 안내 문구
        if not roi_finalized:
            msg = "ROI Setup - L: add point, R: finalize, Enter/Space/S: start, ESC: skip"
        else:
            msg = "ROI fixed - Press Enter/Space/S to start"
        cv2.putText(disp, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

        cv2.imshow(win_name, disp)
        key = cv2.waitKey(20) & 0xFF

        # Enter(13), Space(32), 's'(115) 눌렀을 때 ROI가 확정된 상태면 시작
        if key in (13, 32, ord('s')):
            if roi_finalized:
                logging.info("ROI 설정 완료, 본 영상 처리 시작")
                break
            else:
                logging.info("ROI가 아직 확정되지 않았습니다. 우클릭으로 먼저 확정하세요.")

        # ESC: ROI 없이 진행
        if key == 27:  # ESC
            logging.info("ROI 설정 취소, ROI 없이 진행합니다.")
            roi_points = []
            roi_finalized = False
            break

    cap.release()
    cv2.destroyWindow(win_name)


if __name__ == "__main__":
    main()


# 이미지 저장
# 빨간색 count