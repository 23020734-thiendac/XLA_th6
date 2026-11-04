"""Bài 2 tuần 6: phân loại hình tròn, vuông, tam giác bằng hình dạng và HOG."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import cv2
import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


THU_MUC = Path(__file__).resolve().parent
THU_MUC_DU_LIEU = THU_MUC / "data" / "shapes"
THU_MUC_ANH_MOI = THU_MUC / "data" / "shapes_external"
THU_MUC_KET_QUA = THU_MUC / "output_bai2"
NHAN = ("circle", "square", "triangle")
TEN_HIEN_THI = {"circle": "Tròn", "square": "Vuông", "triangle": "Tam giác"}
DUOI_ANH = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TEN_HINH_DANG = [
    "area",
    "perimeter",
    "width",
    "height",
    "aspect_ratio",
    "circularity",
    "edge_density",
]
KICH_THUOC_HOG = (64, 64)
HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
SO_CHIEU_HOG = int(HOG.getDescriptorSize())


@dataclass(frozen=True)
class ThiNghiem:
    ma: str
    nhom_dac_trung: str
    mo_hinh: str


CAC_THI_NGHIEM = tuple(
    ThiNghiem(f"{nhom}_{mo_hinh}", nhom, mo_hinh)
    for nhom in ("Shape", "HOG", "Combined")
    for mo_hinh in ("KNN", "SVM", "RF")
)


def phan_tich_tham_so() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phân loại tròn, vuông, tam giác bằng Shape/HOG."
    )
    parser.add_argument("--data", type=Path, default=THU_MUC_DU_LIEU)
    parser.add_argument("--data-moi", type=Path, default=THU_MUC_ANH_MOI)
    parser.add_argument("--output", type=Path, default=THU_MUC_KET_QUA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hien-thi", action="store_true")
    return parser.parse_args()


def doc_anh(duong_dan: Path) -> np.ndarray:
    try:
        du_lieu = np.fromfile(duong_dan, dtype=np.uint8)
    except OSError as exc:
        raise FileNotFoundError(f"Không thể đọc ảnh: {duong_dan}") from exc
    anh = cv2.imdecode(du_lieu, cv2.IMREAD_COLOR)
    if anh is None:
        raise ValueError(f"Tệp không phải ảnh hợp lệ: {duong_dan}")
    return anh


def liet_ke_du_lieu(thu_muc: Path, toi_thieu: int) -> list[tuple[Path, str]]:
    ket_qua: list[tuple[Path, str]] = []
    thieu: list[str] = []
    for nhan in NHAN:
        thu_muc_nhan = thu_muc / nhan
        cac_anh = sorted(
            tep
            for tep in thu_muc_nhan.rglob("*")
            if tep.is_file() and tep.suffix.lower() in DUOI_ANH
        ) if thu_muc_nhan.exists() else []
        if len(cac_anh) < toi_thieu:
            thieu.append(f"{nhan}: {len(cac_anh)}")
        ket_qua.extend((tep, nhan) for tep in cac_anh)
    if thieu:
        raise ValueError(
            f"Dữ liệu chưa đủ ({', '.join(thieu)}). "
            "Hãy chạy tao_du_lieu_hinh_hoc.py trước."
        )
    return ket_qua


def tien_xu_ly(anh_bgr: np.ndarray) -> dict[str, np.ndarray | tuple[int, int, int, int]]:
    anh_xam = cv2.cvtColor(anh_bgr, cv2.COLOR_BGR2GRAY)
    anh_min = cv2.GaussianBlur(anh_xam, (5, 5), 0)
    _, nhi_phan = cv2.threshold(
        anh_min, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    gia_tri_goc = np.array(
        [nhi_phan[0, 0], nhi_phan[0, -1], nhi_phan[-1, 0], nhi_phan[-1, -1]]
    )
    if float(gia_tri_goc.mean()) > 127:
        nhi_phan = cv2.bitwise_not(nhi_phan)
    kernel = np.ones((3, 3), dtype=np.uint8)
    nhi_phan = cv2.morphologyEx(nhi_phan, cv2.MORPH_OPEN, kernel)
    nhi_phan = cv2.morphologyEx(nhi_phan, cv2.MORPH_CLOSE, kernel)
    canny = cv2.Canny(anh_min, 50, 150)

    cac_contour, _ = cv2.findContours(
        nhi_phan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not cac_contour:
        raise ValueError("Không tìm thấy contour trong ảnh.")
    contour = max(cac_contour, key=cv2.contourArea)
    if cv2.contourArea(contour) < 50:
        raise ValueError("Contour lớn nhất quá nhỏ, không phải đối tượng hợp lệ.")
    khung_bao = cv2.boundingRect(contour)
    return {
        "gray": anh_xam,
        "blur": anh_min,
        "binary": nhi_phan,
        "canny": canny,
        "contour": contour,
        "bbox": khung_bao,
    }


def chuan_hoa_hog(
    nhi_phan: np.ndarray, khung_bao: tuple[int, int, int, int]
) -> np.ndarray:
    x, y, w, h = khung_bao
    roi = nhi_phan[y : y + h, x : x + w]
    canh = max(w, h) + 10
    canvas = np.zeros((canh, canh), dtype=np.uint8)
    y0 = (canh - h) // 2
    x0 = (canh - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = roi
    return cv2.resize(canvas, KICH_THUOC_HOG, interpolation=cv2.INTER_AREA)


def trich_chon_dac_trung(
    anh_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    tg = tien_xu_ly(anh_bgr)
    contour = tg["contour"]
    x, y, w, h = tg["bbox"]
    dien_tich = float(cv2.contourArea(contour))
    chu_vi = float(cv2.arcLength(contour, True))
    ty_le = float(w / h) if h else 0.0
    do_tron = float(4 * math.pi * dien_tich / (chu_vi * chu_vi)) if chu_vi else 0.0
    canny_roi = tg["canny"][y : y + h, x : x + w]
    mat_do_bien = float(np.count_nonzero(canny_roi) / (w * h)) if w * h else 0.0
    hinh_dang = np.asarray(
        [dien_tich, chu_vi, float(w), float(h), ty_le, do_tron, mat_do_bien],
        dtype=np.float64,
    )

    anh_hog = chuan_hoa_hog(tg["binary"], tg["bbox"])
    vector_hog = HOG.compute(anh_hog).ravel().astype(np.float64)
    tg["hog_input"] = anh_hog
    return hinh_dang, vector_hog, tg


def tao_anh_truc_quan_hog(anh_hog: np.ndarray, he_so: int = 4) -> np.ndarray:
    gx = cv2.Sobel(anh_hog, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(anh_hog, cv2.CV_32F, 0, 1, ksize=1)
    do_lon, goc = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    goc %= 180.0
    kich_thuoc_cell = 8
    so_bin = 9
    anh = np.zeros((64 * he_so, 64 * he_so, 3), dtype=np.uint8)
    for cy in range(8):
        for cx in range(8):
            ys = slice(cy * kich_thuoc_cell, (cy + 1) * kich_thuoc_cell)
            xs = slice(cx * kich_thuoc_cell, (cx + 1) * kich_thuoc_cell)
            hist = np.zeros(so_bin, dtype=np.float32)
            chi_so_bin = np.floor(goc[ys, xs] / 20).astype(int) % so_bin
            for b in range(so_bin):
                hist[b] = float(do_lon[ys, xs][chi_so_bin == b].sum())
            lon_nhat = float(hist.max())
            if lon_nhat <= 0:
                continue
            tam_x = int((cx * 8 + 4) * he_so)
            tam_y = int((cy * 8 + 4) * he_so)
            for b, gia_tri in enumerate(hist):
                chieu_dai = (gia_tri / lon_nhat) * 3.5 * he_so
                goc_rad = math.radians(b * 20 + 10)
                dx = int(round(math.cos(goc_rad) * chieu_dai))
                dy = int(round(math.sin(goc_rad) * chieu_dai))
                cv2.line(
                    anh,
                    (tam_x - dx, tam_y - dy),
                    (tam_x + dx, tam_y + dy),
                    (80, 255, 80),
                    1,
                    cv2.LINE_AA,
                )
    return anh


def tao_mo_hinh(ten: str, seed: int) -> Pipeline:
    if ten == "KNN":
        return Pipeline(
            (("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(5, weights="distance")))
        )
    if ten == "SVM":
        return Pipeline(
            (("scaler", StandardScaler()), ("classifier", SVC(C=10.0, gamma="scale")))
        )
    if ten == "RF":
        return Pipeline(
            (("classifier", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),)
        )
    raise ValueError(f"Mô hình không hợp lệ: {ten}")


def luu_minh_hoa(du_lieu: list[tuple[Path, str]], duong_dan: Path) -> None:
    import matplotlib.pyplot as plt

    dai_dien = {nhan: next(tep for tep, n in du_lieu if n == nhan) for nhan in NHAN}
    fig, truc = plt.subplots(3, 8, figsize=(21, 8.5))
    for hang, nhan in enumerate(NHAN):
        anh = doc_anh(dai_dien[nhan])
        _, _, tg = trich_chon_dac_trung(anh)
        contour_img = anh.copy()
        cv2.drawContours(contour_img, [tg["contour"]], -1, (0, 0, 255), 2)
        hog_vis = tao_anh_truc_quan_hog(tg["hog_input"])
        cac_anh = (
            cv2.cvtColor(anh, cv2.COLOR_BGR2RGB),
            tg["gray"],
            tg["blur"],
            tg["binary"],
            tg["canny"],
            cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB),
            tg["hog_input"],
            cv2.cvtColor(hog_vis, cv2.COLOR_BGR2RGB),
        )
        tieu_de = ("Ảnh gốc", "Grayscale", "Gaussian", "Otsu", "Canny", "Contour", "Chuẩn hóa 64×64", "HOG")
        for cot, (anh_hien_thi, ten) in enumerate(zip(cac_anh, tieu_de, strict=True)):
            truc[hang, cot].imshow(anh_hien_thi, cmap=None if anh_hien_thi.ndim == 3 else "gray")
            truc[hang, cot].set_title(ten, fontsize=10)
            truc[hang, cot].axis("off")
        truc[hang, 0].set_ylabel(TEN_HIEN_THI[nhan], fontsize=12)
    fig.suptitle("Tiền xử lý, phân đoạn, Contour và đặc trưng HOG", fontsize=15)
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=180, bbox_inches="tight")
    plt.close(fig)


def ve_so_sanh(bang: pd.DataFrame, duong_dan: Path) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(bang))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 0.18, bang["accuracy"], 0.36, label="Accuracy")
    ax.bar(x + 0.18, bang["f1_macro"], 0.36, label="F1-macro")
    ax.set_xticks(x, bang["experiment"], rotation=35, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Giá trị")
    ax.set_title("So sánh 9 tổ hợp đặc trưng và mô hình")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=180, bbox_inches="tight")
    plt.close(fig)


def ve_confusion_matrices(
    cac_du_doan: dict[str, np.ndarray], y_test: np.ndarray, duong_dan: Path
) -> None:
    import matplotlib.pyplot as plt

    fig, cac_truc = plt.subplots(3, 3, figsize=(14, 13))
    for ax, tn in zip(cac_truc.ravel(), CAC_THI_NGHIEM, strict=True):
        cm = confusion_matrix(y_test, cac_du_doan[tn.ma], labels=NHAN)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3), [TEN_HIEN_THI[n] for n in NHAN], rotation=20)
        ax.set_yticks(range(3), [TEN_HIEN_THI[n] for n in NHAN])
        ax.set_title(tn.ma)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Nhãn thật")
        nguong = cm.max() / 2
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > nguong else "black")
    fig.suptitle("Confusion Matrix của 9 thí nghiệm", fontsize=16)
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=180, bbox_inches="tight")
    plt.close(fig)


def chon_x(nhom: str, x_shape: np.ndarray, x_hog: np.ndarray) -> np.ndarray:
    if nhom == "Shape":
        return x_shape
    if nhom == "HOG":
        return x_hog
    if nhom == "Combined":
        return np.hstack((x_shape, x_hog))
    raise ValueError(f"Nhóm đặc trưng không hợp lệ: {nhom}")


def luu_du_doan_anh_moi(
    du_lieu_moi: list[tuple[Path, str]],
    mo_hinh: Pipeline,
    nhom_dac_trung: str,
    duong_dan: Path,
) -> tuple[int, int]:
    import matplotlib.pyplot as plt

    anh_va_ket_qua = []
    so_dung = 0
    for tep, nhan in du_lieu_moi:
        anh = doc_anh(tep)
        shape, hog, _ = trich_chon_dac_trung(anh)
        x = chon_x(nhom_dac_trung, shape.reshape(1, -1), hog.reshape(1, -1))
        du_doan = str(mo_hinh.predict(x)[0])
        so_dung += int(du_doan == nhan)
        anh_va_ket_qua.append((anh, nhan, du_doan))

    so_cot = 5
    so_hang = math.ceil(len(anh_va_ket_qua) / so_cot)
    fig, cac_truc = plt.subplots(so_hang, so_cot, figsize=(14, 2.8 * so_hang))
    for ax, (anh, nhan, du_doan) in zip(np.asarray(cac_truc).ravel(), anh_va_ket_qua, strict=False):
        ax.imshow(cv2.cvtColor(anh, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"Thật: {TEN_HIEN_THI[nhan]}\nĐoán: {TEN_HIEN_THI[du_doan]}",
            color="green" if nhan == du_doan else "red",
            fontsize=9,
        )
        ax.axis("off")
    for ax in np.asarray(cac_truc).ravel()[len(anh_va_ket_qua):]:
        ax.axis("off")
    fig.suptitle(f"Dự đoán {len(anh_va_ket_qua)} ảnh mới ngoài tập huấn luyện")
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return so_dung, len(anh_va_ket_qua)


def luu_phan_tich_sai(
    bang: pd.DataFrame,
    chi_so_test: np.ndarray,
    y_test: np.ndarray,
    du_doan: np.ndarray,
    duong_dan: Path,
) -> int:
    import matplotlib.pyplot as plt

    vi_tri_sai = np.flatnonzero(y_test != du_doan)
    if len(vi_tri_sai) == 0:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.text(0.5, 0.58, "Không có mẫu bị phân loại sai", ha="center", va="center", fontsize=18)
        ax.text(0.5, 0.35, "Tập hình học có độ tương phản cao và ba đường bao khác biệt rõ.", ha="center", va="center", fontsize=11)
        ax.axis("off")
        fig.savefig(duong_dan, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return 0

    so_luong = min(12, len(vi_tri_sai))
    fig, cac_truc = plt.subplots(3, 4, figsize=(11, 8.5))
    for ax, vi_tri in zip(cac_truc.ravel(), vi_tri_sai[:so_luong], strict=False):
        hang = bang.iloc[int(chi_so_test[vi_tri])]
        anh = doc_anh(Path(str(hang["path"])))
        ax.imshow(cv2.cvtColor(anh, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Thật: {TEN_HIEN_THI[y_test[vi_tri]]}\nĐoán: {TEN_HIEN_THI[du_doan[vi_tri]]}", color="red")
        ax.axis("off")
    for ax in cac_truc.ravel()[so_luong:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return int(len(vi_tri_sai))


def viet_bao_cao(
    bang_ket_qua: pd.DataFrame,
    so_luong_nhan: dict[str, int],
    so_train: int,
    so_test: int,
    ma_tot_nhat: str,
    so_sai: int,
    so_ngoai_dung: int,
    so_ngoai: int,
    duong_dan: Path,
) -> None:
    cac_dong = [
        f"{h.experiment}: {h.feature_group} + {h.model} | "
        f"Accuracy={h.accuracy:.4f}, F1-macro={h.f1_macro:.4f}"
        for h in bang_ket_qua.itertuples()
    ]
    noi_dung = f"""BÁO CÁO BÀI 2 - ĐẶC TRƯNG HÌNH DẠNG VÀ HOG

1. Dữ liệu
- Số ảnh: {so_luong_nhan}; tổng {sum(so_luong_nhan.values())} ảnh.
- Ảnh 128×128, một vật thể tối trên nền sáng; thay đổi vị trí, kích thước, góc quay, độ sáng và nhiễu.
- Train/Test = {so_train}/{so_test} (80/20, phân tầng, cùng chỉ số chia cho mọi thí nghiệm).
- Có thêm {so_ngoai} ảnh sinh bằng seed khác, không tham gia huấn luyện hay chọn mô hình.

2. Tiền xử lý và đặc trưng
- Grayscale → Gaussian 5×5 → Otsu → morphology → Canny → contour lớn nhất.
- Vector hình dạng: {len(TEN_HINH_DANG)} chiều ({', '.join(TEN_HINH_DANG)}).
- HOG: ảnh chuẩn hóa 64×64, cell 8×8, block 16×16, stride 8×8, 9 bins; vector {SO_CHIEU_HOG} chiều.
- Vector kết hợp: {len(TEN_HINH_DANG) + SO_CHIEU_HOG} chiều.

3. Kết quả 9 thí nghiệm
{chr(10).join(cac_dong)}

Mô hình tốt nhất theo F1-macro (ưu tiên ít chiều nếu hòa): {ma_tot_nhat}.
Ảnh ngoài tập huấn luyện dự đoán đúng: {so_ngoai_dung}/{so_ngoai}.
Số mẫu Test bị phân loại sai bởi mô hình tốt nhất: {so_sai}.

Phân tích lỗi: nếu không có mẫu sai, nguyên nhân là dữ liệu có nền đơn giản, tương phản cao và contour ba lớp khác biệt rõ. Trong ảnh thực tế, lỗi có thể xuất hiện khi vật thể bị che khuất/cắt biên, Otsu tách nền sai, nhiễu tạo contour giả, hình vuông quay và mờ làm mất góc, hoặc tam giác bị bo tròn.

4. Câu hỏi thảo luận
Câu 12. Vì sao đặc trưng hình dạng phân biệt được tròn, vuông và tam giác?
Ba đường bao có quan hệ diện tích–chu vi khác nhau. Độ tròn C=4πA/P² gần 1 với hình tròn, khoảng π/4 với hình vuông lý tưởng và thấp hơn với tam giác đều. Diện tích, chu vi, tỷ lệ khung bao và mật độ biên bổ sung thông tin về kích thước và cấu trúc đường bao.

Câu 13. HOG mô tả thông tin gì?
HOG thống kê hướng và độ lớn gradient trong các ô cục bộ, sau đó chuẩn hóa theo block. Nó mô tả cách các cạnh phân bố trong ảnh, nên giữ được bố cục đường thẳng, góc và đường cong, đồng thời ít nhạy với thay đổi độ sáng đồng đều.

Câu 14. Thay đổi kích thước hoặc góc quay ảnh hưởng ra sao?
Diện tích và chu vi thay đổi theo tỷ lệ ảnh, còn aspect ratio và circularity ổn định hơn. Bước crop–padding–resize giúp HOG bớt nhạy với kích thước và vị trí, nhưng HOG không bất biến quay: khi vật thể quay, các bin hướng gradient thay đổi. Data augmentation nhiều góc quay giúp mô hình học được biến thiên này.

Câu 15. Khi nào nên dùng hình dạng, khi nào nên dùng HOG?
Nên dùng đặc trưng hình dạng khi vật thể phân đoạn rõ, silhouette đơn giản và cần mô hình nhẹ, dễ giải thích. Nên dùng HOG khi bố cục cạnh/góc cục bộ quan trọng hoặc contour đơn lẻ chưa đủ. Có thể kết hợp khi hai nhóm bổ sung nhau, nhưng phải đánh giá trên Test vì số chiều HOG lớn không bảo đảm chính xác hơn.
"""
    duong_dan.write_text(noi_dung, encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = phan_tich_tham_so()
    if not args.hien_thi:
        matplotlib.use("Agg")

    du_lieu = liet_ke_du_lieu(args.data, toi_thieu=50)
    du_lieu_moi = liet_ke_du_lieu(args.data_moi, toi_thieu=1)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Dữ liệu huấn luyện/kiểm thử: {len(du_lieu)} ảnh")

    cac_shape = []
    cac_hog = []
    cac_hang = []
    for i, (tep, nhan) in enumerate(du_lieu, start=1):
        shape, hog, _ = trich_chon_dac_trung(doc_anh(tep))
        cac_shape.append(shape)
        cac_hog.append(hog)
        hang: dict[str, object] = dict(zip(TEN_HINH_DANG, shape, strict=True))
        hang.update({"label": nhan, "path": str(tep.resolve())})
        cac_hang.append(hang)
        if i % 100 == 0 or i == len(du_lieu):
            print(f"Đã trích đặc trưng {i}/{len(du_lieu)} ảnh")

    x_shape = np.vstack(cac_shape)
    x_hog = np.vstack(cac_hog)
    bang = pd.DataFrame(cac_hang)
    y = bang["label"].to_numpy()
    bang.to_csv(args.output / "02_bang_dac_trung_hinh_dang.csv", index=False, encoding="utf-8-sig")
    np.save(args.output / "03_dac_trung_hog.npy", x_hog)
    luu_minh_hoa(du_lieu, args.output / "01_minh_hoa_tien_xu_ly_hog.png")

    chi_so = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        chi_so, test_size=0.20, random_state=args.seed, stratify=y
    )
    y_train, y_test = y[train_idx], y[test_idx]
    cac_ket_qua = []
    cac_du_doan: dict[str, np.ndarray] = {}
    cac_mo_hinh: dict[str, Pipeline] = {}
    for tn in CAC_THI_NGHIEM:
        x = chon_x(tn.nhom_dac_trung, x_shape, x_hog)
        mo_hinh = tao_mo_hinh(tn.mo_hinh, args.seed)
        mo_hinh.fit(x[train_idx], y_train)
        du_doan = mo_hinh.predict(x[test_idx])
        cac_mo_hinh[tn.ma] = mo_hinh
        cac_du_doan[tn.ma] = du_doan
        cac_ket_qua.append(
            {
                "experiment": tn.ma,
                "feature_group": tn.nhom_dac_trung,
                "feature_count": x.shape[1],
                "model": tn.mo_hinh,
                "accuracy": accuracy_score(y_test, du_doan),
                "f1_macro": f1_score(y_test, du_doan, average="macro", zero_division=0),
            }
        )
        print(f"Hoàn tất {tn.ma}")

    bang_ket_qua = pd.DataFrame(cac_ket_qua)
    bang_ket_qua.to_csv(args.output / "04_bang_so_sanh_mo_hinh.csv", index=False, encoding="utf-8-sig")
    ve_so_sanh(bang_ket_qua, args.output / "05_bieu_do_so_sanh.png")
    ve_confusion_matrices(cac_du_doan, y_test, args.output / "06_confusion_matrix.png")

    bang_xep_hang = bang_ket_qua.sort_values(
        ["f1_macro", "accuracy", "feature_count"], ascending=[False, False, True]
    )
    ma_tot_nhat = str(bang_xep_hang.iloc[0]["experiment"])
    tn_tot_nhat = next(tn for tn in CAC_THI_NGHIEM if tn.ma == ma_tot_nhat)
    mo_hinh_tot_nhat = cac_mo_hinh[ma_tot_nhat]
    joblib.dump(
        {
            "pipeline": mo_hinh_tot_nhat,
            "feature_group": tn_tot_nhat.nhom_dac_trung,
            "shape_feature_names": TEN_HINH_DANG,
            "hog_dimension": SO_CHIEU_HOG,
            "labels": NHAN,
        },
        args.output / "07_mo_hinh_tot_nhat.joblib",
    )

    so_ngoai_dung, so_ngoai = luu_du_doan_anh_moi(
        du_lieu_moi,
        mo_hinh_tot_nhat,
        tn_tot_nhat.nhom_dac_trung,
        args.output / "08_du_doan_anh_moi.png",
    )
    so_sai = luu_phan_tich_sai(
        bang,
        test_idx,
        y_test,
        cac_du_doan[ma_tot_nhat],
        args.output / "09_phan_tich_mau_sai.png",
    )
    so_luong_nhan = bang["label"].value_counts().sort_index().to_dict()
    viet_bao_cao(
        bang_ket_qua,
        so_luong_nhan,
        len(train_idx),
        len(test_idx),
        ma_tot_nhat,
        so_sai,
        so_ngoai_dung,
        so_ngoai,
        args.output / "10_bao_cao_va_thao_luan.txt",
    )

    print("\nKẾT QUẢ 9 THÍ NGHIỆM")
    print(bang_ket_qua.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nMô hình tốt nhất: {ma_tot_nhat}")
    print(f"Dự đoán ảnh ngoài tập huấn luyện: {so_ngoai_dung}/{so_ngoai} đúng")
    print(f"Đã lưu kết quả tại: {args.output.resolve()}")

    if args.hien_thi:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
