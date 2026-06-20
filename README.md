# ♻️ Web App Klasifikasi Sampah Berbasis CNN-KAN

Aplikasi Streamlit untuk inferensi model **CNN + Kolmogorov-Arnold Network (KAN)** yang dilatih untuk klasifikasi 12 kategori sampah. Mendukung unggah file, kamera langsung, prediksi batch, dan UI bilingual (Indonesia / English).

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka browser di `http://localhost:8501`.

> Pastikan file model `CNN_KAN.keras` berada di folder yang sama dengan `app.py`.

## 📁 Struktur Proyek

```
garbage_classification/
├── app.py              # Aplikasi Streamlit utama
├── CNN_KAN.keras       # File model (download dari Kaggle / training)
├── requirements.txt    # Dependensi Python
├── pyproject.toml      # Metadata proyek
└── README.md
```

## ✨ Fitur

| Fitur | Deskripsi |
|---|---|
| **Prediksi Tunggal** | Unggah satu gambar atau ambil foto langsung dari kamera |
| **Kamera Real-time** | Ambil foto menggunakan webcam tanpa perlu file eksternal |
| **Confidence Score** | Gauge visual + persentase probabilitas kelas terprediksi |
| **Distribusi Probabilitas** | Progress bar untuk semua 12 kelas, diurutkan dari tertinggi |
| **Prediksi Batch** | Proses banyak gambar sekaligus dengan progress bar dan tabel ringkasan |
| **Metadata Gambar** | Menampilkan dimensi, mode warna, dan nama file |
| **Waktu Inferensi** | Latensi prediksi dalam milidetik |
| **🌐 Bilingual UI** | Bahasa Indonesia dan English, toggle di sidebar |
| **Dark/Light Theme** | Tema terang modern dengan aksen hijau |

## 🏷️ 12 Kelas Sampah

| # | Kelas (ID) | Kelas (EN) | Emoji |
|---|---|---|---|
| 0 | Baterai | Battery | 🔋 |
| 1 | Organik | Organic | 🌿 |
| 2 | Kaca Coklat | Brown Glass | 🟤 |
| 3 | Kardus | Cardboard | 📦 |
| 4 | Pakaian | Clothes | 👕 |
| 5 | Kaca Hijau | Green Glass | 🍶 |
| 6 | Logam | Metal | 🔩 |
| 7 | Kertas | Paper | 📄 |
| 8 | Plastik | Plastic | 🧴 |
| 9 | Sepatu | Shoes | 👟 |
| 10 | Sampah Umum | General Trash | 🗑️ |
| 11 | Kaca Putih | White Glass | 🪟 |

## 🧠 Spesifikasi Model

- **Arsitektur:** CNN (5 Conv Blocks) + 2× DenseKAN Head
- **Input:** 128 × 128 × 3 (RGB, dinormalisasi [0, 1])
- **Output:** Softmax 12 kelas
- **Framework:** TensorFlow 2.16+ + tf-kan-latest
- **Regularisasi:** L2, BatchNorm, SpatialDropout2D, Dropout
- **Augmentasi (training):** RandomFlip, RandomRotation, RandomTranslation, RandomZoom

## 📦 Dependensi

```
streamlit>=1.35.0
tensorflow>=2.16.0
tf-kan-latest>=1.1.0
numpy>=1.26.0
Pillow>=10.0.0
pandas>=2.0.0
```

## 🔧 Konfigurasi

Ubah konstanta di `app.py` sesuai kebutuhan:

```python
IMAGE_SIZE = (128, 128)          # Ukuran input model
MODEL_PATH = "CNN_KAN.keras"     # Path file model
CONFIDENCE_THRESHOLD = 0.5       # Ambang batas confidence warning
```

## 📄 Lisensi

Aplikasi ini dilisensikan di bawah Apache License 2.0. Lihat [LICENSE](LICENSE) untuk detail.