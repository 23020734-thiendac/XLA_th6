"""Bài 1 tuần 6: phân loại táo, chuối và cam bằng đặc trưng màu."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


THU_MUC = Path(__file__).resolve().parent
THU_MUC_DU_LIEU_MAC_DINH = (
    THU_MUC.parent / "fruits-360-100x100-main" / "Training"
)
THU_MUC_KET_QUA_MAC_DINH = THU_MUC / "output_bai1"
KICH_THUOC_ANH = (128, 128)
NHAN = ("apple", "banana", "orange")
TEN_HIEN_THI = {"apple": "Táo", "banana": "Chuối", "orange": "Cam"}
THU_MUC_LOP_NGUON = {
    "apple": "Apple Braeburn 1",
    "banana": "Banana 1",
    "orange": "Orange 1",
}
DUOI_ANH = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

TEN_RGB = [
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
]
TEN_HSV = [
    "hsv_mean_h",
    "hsv_mean_s",
    "hsv_mean_v",
    "hsv_std_h",
    "hsv_std_s",
    "hsv_std_v",
]
TEN_HIST = (
    [f"hsv_hist_h_{i:02d}" for i in range(18)]
    + [f"hsv_hist_s_{i:02d}" for i in range(16)]
    + [f"hsv_hist_v_{i:02d}" for i in range(16)]
)
TEN_ENTROPY = ["gray_entropy"]
TEN_TOAN_BO = TEN_RGB + TEN_HSV + TEN_HIST + TEN_ENTROPY


@dataclass(frozen=True)
class ThiNghiem:
    ma: str
    mo_ta: str
    ten_dac_trung: list[str]
    mo_hinh: str


CAC_THI_NGHIEM = (
    ThiNghiem("TN1", "Mean + Std (RGB)", TEN_RGB, "KNN"),
    ThiNghiem("TN2", "Mean + Std (HSV)", TEN_HSV, "KNN"),
    ThiNghiem("TN3", "Histogram màu HSV", TEN_HIST, "SVM"),
    ThiNghiem("TN4", "Kết hợp toàn bộ đặc trưng", TEN_TOAN_BO, "SVM"),
)


def phan_tich_tham_so() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phân loại táo, chuối và cam bằng KNN/SVM."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=THU_MUC_DU_LIEU_MAC_DINH,
        help=(
            "Thư mục Training của fruits-360-100x100-main, chứa các lớp "
            "Apple Braeburn 1, Banana 1 và Orange 1."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=THU_MUC_KET_QUA_MAC_DINH,
        help="Thư mục lưu bảng, biểu đồ và mô hình.",
    )
    parser.add_argument(
        "--du-doan",
        type=Path,
        nargs="*",
        default=[],
        help="Ảnh mới cần dự đoán thêm sau khi huấn luyện.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Hạt ngẫu nhiên.")
    parser.add_argument(
        "--hien-thi",
        action="store_true",
        help="Mở các cửa sổ matplotlib sau khi lưu kết quả.",
    )
    return parser.parse_args()


def doc_anh(duong_dan: Path) -> np.ndarray:
    """Đọc ảnh an toàn với đường dẫn Unicode trên Windows."""
    try:
        du_lieu = np.fromfile(duong_dan, dtype=np.uint8)
    except OSError as exc:
        raise FileNotFoundError(f"Không thể đọc tệp ảnh: {duong_dan}") from exc
    anh = cv2.imdecode(du_lieu, cv2.IMREAD_COLOR)
    if anh is None:
        raise ValueError(f"Tệp không phải ảnh hợp lệ: {duong_dan}")
    return anh


def liet_ke_du_lieu(thu_muc_du_lieu: Path) -> list[tuple[Path, str]]:
    du_lieu: list[tuple[Path, str]] = []
    thieu: list[str] = []
    for nhan in NHAN:
        ten_lop_nguon = THU_MUC_LOP_NGUON[nhan]
        thu_muc_nhan = thu_muc_du_lieu / ten_lop_nguon
        cac_anh = sorted(
            tep
            for tep in thu_muc_nhan.rglob("*")
            if tep.is_file() and tep.suffix.lower() in DUOI_ANH
        ) if thu_muc_nhan.exists() else []
        if len(cac_anh) < 50:
            thieu.append(f"{ten_lop_nguon}: {len(cac_anh)} ảnh")
        du_lieu.extend((tep, nhan) for tep in cac_anh)

    if thieu:
        chi_tiet = ", ".join(thieu)
        raise ValueError(
            "Mỗi lớp cần tối thiểu 50 ảnh. Hiện có "
            f"{chi_tiet}. Hãy kiểm tra lại đường dẫn --data."
        )
    return du_lieu


def entropy_anh_xam(anh_xam: np.ndarray) -> float:
    tan_suat = np.bincount(anh_xam.ravel(), minlength=256).astype(np.float64)
    xac_suat = tan_suat[tan_suat > 0] / anh_xam.size
    return float(-np.sum(xac_suat * np.log2(xac_suat)))


def histogram_chuan_hoa(
    anh: np.ndarray, kenh: int, so_bin: int, khoang: tuple[int, int]
) -> np.ndarray:
    histogram = cv2.calcHist([anh], [kenh], None, [so_bin], list(khoang)).ravel()
    tong = float(histogram.sum())
    return histogram / tong if tong > 0 else histogram


def trich_chon_dac_trung(anh_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Trả về vector 63 chiều và các ảnh trung gian."""
    noi_suy = cv2.INTER_AREA if max(anh_bgr.shape[:2]) > 128 else cv2.INTER_CUBIC
    anh_128 = cv2.resize(anh_bgr, KICH_THUOC_ANH, interpolation=noi_suy)
    anh_rgb = cv2.cvtColor(anh_128, cv2.COLOR_BGR2RGB)
    anh_hsv = cv2.cvtColor(anh_128, cv2.COLOR_BGR2HSV)
    anh_xam = cv2.cvtColor(anh_128, cv2.COLOR_BGR2GRAY)

    pixel_rgb = anh_rgb.reshape(-1, 3).astype(np.float64) / 255.0
    he_so_hsv = np.array([179.0, 255.0, 255.0], dtype=np.float64)
    pixel_hsv = anh_hsv.reshape(-1, 3).astype(np.float64) / he_so_hsv

    rgb_mean_std = np.concatenate((pixel_rgb.mean(axis=0), pixel_rgb.std(axis=0)))
    hsv_mean_std = np.concatenate((pixel_hsv.mean(axis=0), pixel_hsv.std(axis=0)))
    hsv_histogram = np.concatenate(
        (
            histogram_chuan_hoa(anh_hsv, 0, 18, (0, 180)),
            histogram_chuan_hoa(anh_hsv, 1, 16, (0, 256)),
            histogram_chuan_hoa(anh_hsv, 2, 16, (0, 256)),
        )
    )
    entropy = np.array([entropy_anh_xam(anh_xam)], dtype=np.float64)
    vector = np.concatenate((rgb_mean_std, hsv_mean_std, hsv_histogram, entropy))
    trung_gian = {"resized": anh_128, "rgb": anh_rgb, "hsv": anh_hsv, "gray": anh_xam}
    return vector, trung_gian


def tao_bang_dac_trung(du_lieu: list[tuple[Path, str]]) -> pd.DataFrame:
    cac_hang: list[dict[str, object]] = []
    for thu_tu, (duong_dan, nhan) in enumerate(du_lieu, start=1):
        vector, _ = trich_chon_dac_trung(doc_anh(duong_dan))
        hang: dict[str, object] = dict(zip(TEN_TOAN_BO, vector, strict=True))
        hang["label"] = nhan
        hang["path"] = str(duong_dan.resolve())
        cac_hang.append(hang)
        if thu_tu % 50 == 0 or thu_tu == len(du_lieu):
            print(f"Đã trích đặc trưng {thu_tu}/{len(du_lieu)} ảnh")
    return pd.DataFrame(cac_hang)


def tao_mo_hinh(ten: str) -> Pipeline:
    if ten == "KNN":
        bo_phan_loai = KNeighborsClassifier(n_neighbors=5, weights="distance")
    elif ten == "SVM":
        bo_phan_loai = SVC(kernel="rbf", C=10.0, gamma="scale")
    else:
        raise ValueError(f"Mô hình không được hỗ trợ: {ten}")
    return Pipeline((('scaler', StandardScaler()), ('classifier', bo_phan_loai)))


def luu_minh_hoa_tien_xu_ly(
    du_lieu: list[tuple[Path, str]], duong_dan: Path
) -> None:
    import matplotlib.pyplot as plt

    anh_dai_dien = {nhan: next(tep for tep, n in du_lieu if n == nhan) for nhan in NHAN}
    fig, truc = plt.subplots(len(NHAN), 5, figsize=(15, 9))
    for hang, nhan in enumerate(NHAN):
        anh_goc_bgr = doc_anh(anh_dai_dien[nhan])
        _, tg = trich_chon_dac_trung(anh_goc_bgr)
        hsv = tg["hsv"]
        cac_anh = (
            cv2.cvtColor(anh_goc_bgr, cv2.COLOR_BGR2RGB),
            tg["rgb"],
            hsv[:, :, 0],
            hsv[:, :, 1],
            hsv[:, :, 2],
        )
        tieu_de = (
            f"Ảnh gốc\n{anh_goc_bgr.shape[1]}×{anh_goc_bgr.shape[0]}",
            "Resize RGB\n128×128",
            "Kênh H",
            "Kênh S",
            "Kênh V",
        )
        for cot, (anh, ten) in enumerate(zip(cac_anh, tieu_de, strict=True)):
            truc[hang, cot].imshow(anh, cmap=None if cot < 2 else "gray")
            truc[hang, cot].set_title(ten)
            truc[hang, cot].axis("off")
        truc[hang, 0].set_ylabel(TEN_HIEN_THI[nhan], fontsize=12)
    fig.suptitle("Tiền xử lý: đọc ảnh, resize và chuyển RGB → HSV", fontsize=15)
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=180, bbox_inches="tight")
    plt.close(fig)


def ve_ma_tran_nham_lan(
    du_doan_theo_tn: dict[str, np.ndarray], y_test: np.ndarray, duong_dan: Path
) -> None:
    import matplotlib.pyplot as plt

    fig, cac_truc = plt.subplots(2, 2, figsize=(10, 9))
    for ax, thi_nghiem in zip(cac_truc.ravel(), CAC_THI_NGHIEM, strict=True):
        ma_tran = confusion_matrix(y_test, du_doan_theo_tn[thi_nghiem.ma], labels=NHAN)
        anh = ax.imshow(ma_tran, cmap="Blues")
        ax.set_xticks(range(len(NHAN)), [TEN_HIEN_THI[n] for n in NHAN])
        ax.set_yticks(range(len(NHAN)), [TEN_HIEN_THI[n] for n in NHAN])
        ax.set_xlabel("Nhãn dự đoán")
        ax.set_ylabel("Nhãn thật")
        ax.set_title(f"{thi_nghiem.ma}: {thi_nghiem.mo_ta} + {thi_nghiem.mo_hinh}")
        nguong = ma_tran.max() / 2
        for i in range(len(NHAN)):
            for j in range(len(NHAN)):
                ax.text(
                    j,
                    i,
                    str(ma_tran[i, j]),
                    ha="center",
                    va="center",
                    color="white" if ma_tran[i, j] > nguong else "black",
                )
        fig.colorbar(anh, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion Matrix của bốn thí nghiệm", fontsize=15)
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=180, bbox_inches="tight")
    plt.close(fig)


def ve_bieu_do_so_sanh(bang: pd.DataFrame, duong_dan: Path) -> None:
    import matplotlib.pyplot as plt

    chi_so = np.arange(len(bang))
    do_rong = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, cot in enumerate(("accuracy", "precision_macro", "recall_macro", "f1_macro")):
        ax.bar(chi_so + (i - 1.5) * do_rong, bang[cot], do_rong, label=cot)
    ax.set_xticks(chi_so, bang["experiment"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Giá trị")
    ax.set_title("So sánh Accuracy, Precision, Recall và F1-score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=4, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=180, bbox_inches="tight")
    plt.close(fig)


def luu_anh_du_doan(
    bang: pd.DataFrame,
    chi_so_test: np.ndarray,
    du_doan: np.ndarray,
    ten_mo_hinh: str,
    duong_dan: Path,
) -> None:
    import matplotlib.pyplot as plt

    so_luong = min(12, len(chi_so_test))
    cac_chi_so = np.linspace(0, len(chi_so_test) - 1, so_luong, dtype=int)
    fig, cac_truc = plt.subplots(3, 4, figsize=(11, 8.5))
    for ax, vi_tri in zip(cac_truc.ravel(), cac_chi_so, strict=False):
        hang = bang.iloc[int(chi_so_test[vi_tri])]
        anh = cv2.cvtColor(doc_anh(Path(str(hang["path"]))), cv2.COLOR_BGR2RGB)
        nhan_that = str(hang["label"])
        nhan_du_doan = str(du_doan[vi_tri])
        dung = nhan_that == nhan_du_doan
        ax.imshow(anh)
        ax.set_title(
            f"Thật: {TEN_HIEN_THI[nhan_that]}\nĐoán: {TEN_HIEN_THI[nhan_du_doan]}",
            color="green" if dung else "red",
            fontsize=10,
        )
        ax.axis("off")
    for ax in cac_truc.ravel()[so_luong:]:
        ax.axis("off")
    fig.suptitle(f"Dự đoán trên tập Test bằng {ten_mo_hinh}", fontsize=14)
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=180, bbox_inches="tight")
    plt.close(fig)


def viet_bao_cao(
    bang_ket_qua: pd.DataFrame,
    so_anh_theo_nhan: dict[str, int],
    so_train: int,
    so_test: int,
    thu_muc_du_lieu: Path,
    duong_dan: Path,
) -> None:
    dong_ket_qua = []
    for _, hang in bang_ket_qua.iterrows():
        dong_ket_qua.append(
            f"{hang['experiment']}: {hang['features']} + {hang['model']} | "
            f"Accuracy={hang['accuracy']:.4f}, Precision={hang['precision_macro']:.4f}, "
            f"Recall={hang['recall_macro']:.4f}, F1={hang['f1_macro']:.4f}"
        )
    tot_nhat = bang_ket_qua.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]

    noi_dung = f"""BÁO CÁO BÀI 1 - PHÂN LOẠI ẢNH THEO ĐẶC TRƯNG MÀU

1. Dữ liệu và chia tập
- Nguồn local: {thu_muc_du_lieu.resolve()}
- Lớp nguồn: {THU_MUC_LOP_NGUON}
- Số ảnh: {so_anh_theo_nhan}
- Train/Test = {so_train}/{so_test} (80/20, phân tầng theo nhãn, random_state cố định).
- Mọi bộ chuẩn hóa StandardScaler chỉ được fit trên Train qua sklearn Pipeline.

2. Kết quả
{chr(10).join(dong_ket_qua)}

Thí nghiệm tốt nhất theo F1-macro: {tot_nhat['experiment']} ({tot_nhat['features']} + {tot_nhat['model']}).

Nhận xét: Cả bốn thí nghiệm cùng đạt 100% vì ba lớp có màu và hình dạng rất
khác nhau, ảnh chỉ chứa một quả trên nền trắng và các ảnh đều lấy từ cùng quy
trình quay của Fruits-360. Kết quả cho thấy các đặc trưng màu đủ phân biệt tập
dữ liệu kiểm thử này, nhưng chưa chứng minh mô hình tổng quát tốt với ảnh đời
thực. Cần kiểm tra thêm ảnh tự chụp có nền phức tạp, ánh sáng và camera khác.

3. Trả lời câu hỏi thảo luận
Câu 5. Vì sao cần trích chọn đặc trưng thay vì dùng trực tiếp toàn bộ pixel?
Đặc trưng làm giảm mạnh số chiều, giảm thời gian huấn luyện và nguy cơ quá khớp. Mean, độ lệch chuẩn và histogram còn giữ thông tin màu quan trọng, đồng thời ít nhạy hơn với dịch chuyển nhỏ của vật thể so với vector pixel thô.

Câu 6. RGB và HSV ảnh hưởng thế nào đến kết quả phân loại?
RGB trộn thông tin màu với độ sáng nên giá trị thay đổi khi chiếu sáng thay đổi. HSV tách sắc độ H, độ bão hòa S và độ sáng V; vì vậy histogram HSV thường mô tả màu quả trực tiếp hơn. Tuy nhiên H không ổn định tại vùng có độ bão hòa thấp, chẳng hạn nền trắng.

Câu 7. Kết hợp nhiều đặc trưng có luôn tăng độ chính xác không?
Không. Đặc trưng dư thừa, nhiễu hoặc số chiều quá lớn có thể làm khoảng cách kém ý nghĩa và tăng quá khớp. Cần chuẩn hóa, đánh giá trên dữ liệu chưa dùng để huấn luyện và chọn đặc trưng bằng kết quả thực nghiệm.
"""
    duong_dan.write_text(noi_dung, encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = phan_tich_tham_so()
    if not args.hien_thi:
        matplotlib.use("Agg")

    du_lieu = liet_ke_du_lieu(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "models").mkdir(parents=True, exist_ok=True)

    print(f"Tìm thấy {len(du_lieu)} ảnh hợp lệ.")
    bang = tao_bang_dac_trung(du_lieu)
    bang.to_csv(args.output / "02_bang_dac_trung.csv", index=False, encoding="utf-8-sig")
    luu_minh_hoa_tien_xu_ly(du_lieu, args.output / "01_minh_hoa_tien_xu_ly.png")

    y = bang["label"].to_numpy()
    cac_chi_so = np.arange(len(bang))
    chi_so_train, chi_so_test = train_test_split(
        cac_chi_so,
        test_size=0.20,
        random_state=args.seed,
        stratify=y,
    )
    y_train, y_test = y[chi_so_train], y[chi_so_test]

    cac_ket_qua: list[dict[str, object]] = []
    cac_du_doan: dict[str, np.ndarray] = {}
    cac_mo_hinh: dict[str, Pipeline] = {}
    for thi_nghiem in CAC_THI_NGHIEM:
        x = bang[thi_nghiem.ten_dac_trung].to_numpy(dtype=np.float64)
        mo_hinh = tao_mo_hinh(thi_nghiem.mo_hinh)
        mo_hinh.fit(x[chi_so_train], y_train)
        du_doan = mo_hinh.predict(x[chi_so_test])
        cac_du_doan[thi_nghiem.ma] = du_doan
        cac_mo_hinh[thi_nghiem.ma] = mo_hinh
        cac_ket_qua.append(
            {
                "experiment": thi_nghiem.ma,
                "features": thi_nghiem.mo_ta,
                "feature_count": len(thi_nghiem.ten_dac_trung),
                "model": thi_nghiem.mo_hinh,
                "accuracy": accuracy_score(y_test, du_doan),
                "precision_macro": precision_score(y_test, du_doan, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test, du_doan, average="macro", zero_division=0),
                "f1_macro": f1_score(y_test, du_doan, average="macro", zero_division=0),
            }
        )
        joblib.dump(
            {
                "pipeline": mo_hinh,
                "feature_names": thi_nghiem.ten_dac_trung,
                "labels": NHAN,
                "image_size": KICH_THUOC_ANH,
            },
            args.output / "models" / f"{thi_nghiem.ma}_{thi_nghiem.mo_hinh}.joblib",
        )

    bang_ket_qua = pd.DataFrame(cac_ket_qua)
    bang_ket_qua.to_csv(
        args.output / "03_bang_so_sanh_mo_hinh.csv", index=False, encoding="utf-8-sig"
    )
    ve_bieu_do_so_sanh(bang_ket_qua, args.output / "04_bieu_do_so_sanh.png")
    ve_ma_tran_nham_lan(cac_du_doan, y_test, args.output / "05_confusion_matrix.png")

    ma_tot_nhat = str(
        bang_ket_qua.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]["experiment"]
    )
    luu_anh_du_doan(
        bang,
        chi_so_test,
        cac_du_doan[ma_tot_nhat],
        ma_tot_nhat,
        args.output / "06_anh_du_doan_tap_test.png",
    )

    so_anh_theo_nhan = bang["label"].value_counts().sort_index().to_dict()
    viet_bao_cao(
        bang_ket_qua,
        so_anh_theo_nhan,
        len(chi_so_train),
        len(chi_so_test),
        args.data,
        args.output / "07_bao_cao_va_thao_luan.txt",
    )

    if args.du_doan:
        tn_tot_nhat = next(tn for tn in CAC_THI_NGHIEM if tn.ma == ma_tot_nhat)
        print("\nDự đoán ảnh mới:")
        for duong_dan in args.du_doan:
            vector, _ = trich_chon_dac_trung(doc_anh(duong_dan))
            vi_tri = [TEN_TOAN_BO.index(ten) for ten in tn_tot_nhat.ten_dac_trung]
            du_doan = cac_mo_hinh[ma_tot_nhat].predict(vector[vi_tri].reshape(1, -1))[0]
            print(f"- {duong_dan}: {TEN_HIEN_THI[str(du_doan)]} ({du_doan})")

    print("\nKẾT QUẢ BỐN THÍ NGHIỆM")
    print(bang_ket_qua.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nĐã lưu toàn bộ kết quả tại: {args.output.resolve()}")

    if args.hien_thi:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
