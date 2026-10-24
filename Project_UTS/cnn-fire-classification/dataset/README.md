# Dataset

Project menggunakan **Synthetic Image fire and non-fire dataset** dari Mendeley Data.

- Halaman: https://data.mendeley.com/datasets/mx33vrmxrv/1
- DOI: https://doi.org/10.17632/mx33vrmxrv.1
- Isi menurut deskripsi sumber: 1.000 gambar sintetis (500 Fire, 500 Non-Fire), resolusi 260 × 260.
- Ukuran arsip pada halaman sumber: sekitar 89,5 MB.
- Lisensi pada sumber: CC BY 4.0.

Unduh otomatis dari root repository:

```powershell
python download_dataset.py
```

Script menyimpan arsip di `data/synthetic_fire_non_fire.zip` dan mengekstrak ke `data/raw/`. Data lokal diabaikan oleh Git. Cantumkan atribusi dataset sesuai lisensi. Karena gambar sintetis, hasil tidak otomatis mewakili kondisi nyata.
