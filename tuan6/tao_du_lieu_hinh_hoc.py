"""Tạo dữ liệu hình tròn, hình vuông và tam giác cho Bài 2 tuần 6."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import cv2
import numpy as np


THU_MUC = Path(__file__).resolve().parent
THU_MUC_DU_LIEU = THU_MUC / "data" / "shapes"
THU_MUC_ANH_MOI = THU_MUC / "data" / "shapes_external"
KICH_THUOC_ANH = 128
CAC_NHAN = ("circle", "square", "triangle")


def phan_tich_tham_so() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tạo ảnh hình học có thay đổi vị trí, kích thước và góc quay."
    )
    parser.add_argument("--output", type=Path, default=THU_MUC_DU_LIEU)
    parser.add_argument("--output-moi", type=Path, default=THU_MUC_ANH_MOI)
    parser.add_argument("--so-anh-moi-lop", type=int, default=300)
    parser.add_argument("--so-anh-moi-lop-ngoai", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def luu_anh(duong_dan: Path, anh: np.ndarray) -> None:
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    thanh_cong, du_lieu = cv2.imencode(duong_dan.suffix or ".png", anh)
    if not thanh_cong:
        raise OSError(f"Không thể mã hóa ảnh: {duong_dan}")
    du_lieu.tofile(duong_dan)


def tao_da_giac_deu(
    tam: tuple[int, int], ban_kinh: float, so_dinh: int, goc_do: float
) -> np.ndarray:
    cac_diem = []
    for i in range(so_dinh):
        goc = math.radians(goc_do + 360.0 * i / so_dinh)
        cac_diem.append(
            [
                round(tam[0] + ban_kinh * math.cos(goc)),
                round(tam[1] + ban_kinh * math.sin(goc)),
            ]
        )
    return np.asarray(cac_diem, dtype=np.int32)


def tao_mot_anh(
    nhan: str, rng: np.random.Generator, nhieu_them: float = 0.0
) -> tuple[np.ndarray, dict[str, object]]:
    nen = int(rng.integers(235, 256))
    mau_vat = int(rng.integers(0, 46))
    anh = np.full((KICH_THUOC_ANH, KICH_THUOC_ANH, 3), nen, dtype=np.uint8)
    goc_quay = float(rng.uniform(0, 360))

    if nhan == "circle":
        ban_kinh = int(rng.integers(21, 43))
        le = ban_kinh + 5
        tam = (
            int(rng.integers(le, KICH_THUOC_ANH - le)),
            int(rng.integers(le, KICH_THUOC_ANH - le)),
        )
        cv2.circle(anh, tam, ban_kinh, (mau_vat,) * 3, thickness=-1, lineType=cv2.LINE_AA)
        kich_thuoc = ban_kinh
    elif nhan == "square":
        nua_canh = float(rng.uniform(22, 39))
        ban_kinh_bao = math.ceil(nua_canh * math.sqrt(2))
        le = ban_kinh_bao + 5
        tam = (
            int(rng.integers(le, KICH_THUOC_ANH - le)),
            int(rng.integers(le, KICH_THUOC_ANH - le)),
        )
        diem = tao_da_giac_deu(tam, nua_canh * math.sqrt(2), 4, goc_quay + 45)
        cv2.fillPoly(anh, [diem], (mau_vat,) * 3, lineType=cv2.LINE_AA)
        kich_thuoc = round(nua_canh * 2, 2)
    elif nhan == "triangle":
        ban_kinh = float(rng.uniform(31, 49))
        le = math.ceil(ban_kinh) + 5
        tam = (
            int(rng.integers(le, KICH_THUOC_ANH - le)),
            int(rng.integers(le, KICH_THUOC_ANH - le)),
        )
        diem = tao_da_giac_deu(tam, ban_kinh, 3, goc_quay - 90)
        cv2.fillPoly(anh, [diem], (mau_vat,) * 3, lineType=cv2.LINE_AA)
        kich_thuoc = round(ban_kinh, 2)
    else:
        raise ValueError(f"Nhãn không hợp lệ: {nhan}")

    sigma = float(rng.uniform(1.5, 5.0 + nhieu_them))
    nhieu = rng.normal(0, sigma, anh.shape).astype(np.float32)
    anh = np.clip(anh.astype(np.float32) + nhieu, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        anh = cv2.GaussianBlur(anh, (3, 3), float(rng.uniform(0.4, 1.0)))

    thong_tin = {
        "label": nhan,
        "center_x": tam[0],
        "center_y": tam[1],
        "size_parameter": kich_thuoc,
        "rotation_degree": round(goc_quay, 3),
        "background_value": nen,
        "object_value": mau_vat,
        "noise_sigma": round(sigma, 3),
    }
    return anh, thong_tin


def tao_tap_du_lieu(
    thu_muc: Path,
    so_anh_moi_lop: int,
    seed: int,
    tien_to: str,
    nhieu_them: float = 0.0,
) -> None:
    rng = np.random.default_rng(seed)
    cac_hang: list[dict[str, object]] = []
    for nhan in CAC_NHAN:
        for i in range(so_anh_moi_lop):
            anh, thong_tin = tao_mot_anh(nhan, rng, nhieu_them=nhieu_them)
            ten_tep = f"{tien_to}_{nhan}_{i:04d}.png"
            duong_dan = thu_muc / nhan / ten_tep
            luu_anh(duong_dan, anh)
            thong_tin["path"] = str(duong_dan.resolve())
            cac_hang.append(thong_tin)
        print(f"Đã tạo {so_anh_moi_lop} ảnh {nhan}")

    thu_muc.mkdir(parents=True, exist_ok=True)
    with (thu_muc / "metadata.csv").open("w", newline="", encoding="utf-8-sig") as tep:
        writer = csv.DictWriter(tep, fieldnames=list(cac_hang[0]))
        writer.writeheader()
        writer.writerows(cac_hang)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = phan_tich_tham_so()
    if args.so_anh_moi_lop < 50:
        raise ValueError("--so-anh-moi-lop phải >= 50 theo yêu cầu đề bài.")
    if args.so_anh_moi_lop_ngoai < 1:
        raise ValueError("--so-anh-moi-lop-ngoai phải >= 1.")

    tao_tap_du_lieu(
        args.output, args.so_anh_moi_lop, args.seed, tien_to="train"
    )
    tao_tap_du_lieu(
        args.output_moi,
        args.so_anh_moi_lop_ngoai,
        args.seed + 100_000,
        tien_to="external",
        nhieu_them=2.0,
    )
    print(
        f"Hoàn tất: {3 * args.so_anh_moi_lop} ảnh dữ liệu tại {args.output.resolve()}"
    )
    print(
        f"Ảnh kiểm tra ngoài tập huấn luyện: {3 * args.so_anh_moi_lop_ngoai} "
        f"ảnh tại {args.output_moi.resolve()}"
    )


if __name__ == "__main__":
    main()
