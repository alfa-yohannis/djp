# Pemecahan Masalah

Gejala, penyebab, dan penanganannya. Mulailah selalu dari:

```powershell
.\.venv\Scripts\python.exe main.py status
```

---

## Indexing

### "Direktori sumber tidak ditemukan"

Shared drive Google belum termuat. Buka Google Drive for Desktop dan pastikan
`G:\Shared drives\Tim Borang M.TI\` dapat diakses dari Explorer, lalu ulangi.

Bila huruf drive Anda berbeda, sesuaikan `SOURCE_DIR` pada `.env`.

### Indexing sangat lambat

Periksa apakah embedding benar-benar berjalan di GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Bila `False`, yang terpasang adalah PyTorch versi CPU. Pasang ulang:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Banyak dokumen berstatus "Kosong"

Halaman tersebut adalah hasil pindai tanpa lapisan teks — lazim untuk SK dan
sertifikat. Jumlahnya dilaporkan di panel samping. Untuk membacanya:

```powershell
.\.venv\Scripts\python.exe -m pip install rapidocr-onnxruntime
```

lalu setel `OCR_ENABLED=1` pada `.env` dan jalankan `main.py ingest --rebuild`.

### Dokumen berstatus "Gagal"

Buka tab **Daftar Dokumen** dan saring ke status *Gagal*; pesan galat per berkas
ditampilkan di bawah nama dokumennya. Penyebab yang lazim:

| Pesan | Penyebab |
|---|---|
| `Tidak ada extractor untuk ekstensi '.doc'` | Format Office lama; konversi ke `.docx` |
| `PackageNotFoundError` | Berkas rusak atau sebenarnya bukan format itu |
| `PermissionError` | Berkas sedang dikunci Excel/Word; tutup dan ulangi |

Satu berkas yang gagal tidak pernah menghentikan keseluruhan proses.

### Indexing terhenti di tengah jalan

Jalankan ulang `main.py ingest`. Sistem membandingkan katalog dengan arsip
vektor, mengenali dokumen yang kehilangan vektor, dan membacanya ulang. Tidak
ada langkah manual yang diperlukan.

### Berkas di `zdev` tidak terindeks

Memang disengaja — `zdev` berisi skrip pengembangan. Materi yang berguna di
dalamnya (pemetaan dosen ke SINTA dan Google Scholar) sudah disalin ke
`corpus-lokal` dan diindeks dari sana.

Bila ada berkas lain di `zdev` yang Anda perlukan, salin ke `corpus-lokal` lalu
jalankan `main.py ingest`. **Jangan** memindahkannya — dokumen di shared drive
harus tetap utuh untuk rekan kerja lain.

---

## Pencarian

### Jawaban selalu "Informasi tidak ditemukan"

Periksa berurutan:

1. **Apakah indeks berisi?** `main.py status` — bila `Dokumen terindeks: 0`,
   jalankan `main.py ingest`.
2. **Apakah `Vektor` bernilai 0** padahal potongan ada? Arsip vektor belum
   tersimpan. Jalankan `main.py ingest`; rekonsiliasi akan memperbaikinya.
3. **Apakah penyaring folder sedang aktif?** Kembalikan ke "Semua folder".
4. **Coba tab Telusuri Potongan.** Bila di sana ada hasil tetapi jawabannya
   tetap kosong, masalahnya ada pada model, bukan pada pencarian.

### Hasil pencarian tidak relevan

- Gunakan istilah resmi yang dipakai dokumen (`EWMP`, `DTPR`, `PkM`, `CPL`).
- Naikkan jumlah sumber dengan penggeser di panel samping.
- Pastikan `RERANK_ENABLED=1`; tanpa reranker, presisi turun cukup terasa.
- Untuk nomor SK atau NIDN, ketik angkanya persis — pencarian kata kunci yang
  akan menemukannya.

### "Dimensi vektor tidak cocok"

Model embedding berganti setelah indeks dibangun. Jalankan:

```powershell
.\.venv\Scripts\python.exe main.py ingest --rebuild
```

---

## Model bahasa

### "Ollama tidak dapat dihubungi"

```powershell
ollama list                      # apakah layanannya merespons?
Get-Service -Name "*ollama*"     # apakah layanannya berjalan?
```

Bila belum berjalan, buka aplikasi Ollama sekali dari Start Menu; setelah itu
layanannya otomatis aktif saat boot.

### "Model belum diunduh"

```powershell
ollama pull qwen3:8b
```

### Jawaban lokal lambat

Wajar bila model baru dimuat pertama kali (beberapa detik). Bila setiap
pertanyaan lambat:

- Buka Task Manager → Performance → GPU dan lihat **Shared GPU memory**. Bila
  angkanya jauh di atas ±0,5 GB saat model dimuat, kartu grafis kepenuhan dan
  Windows meminjam RAM sistem; model bisa melambat sampai sepuluh kali tanpa
  pesan galat. `nvidia-smi` tidak menampilkan hal ini.
- Pastikan `QUERY_EMBED_DEVICE=cpu` (bawaan), agar BGE-M3 tidak ikut menempati
  GPU saat melayani pertanyaan.
- Tutup aplikasi lain yang memakai GPU (game, penyuntingan video, peramban
  dengan banyak tab video).
- Bila masih sempit, aktifkan cache KV 8-bit pada layanan Ollama
  (`OLLAMA_KV_CACHE_TYPE=q8_0`, lalu mulai ulang Ollama): menghemat ±0,5 GB.
  Bila memori sudah cukup, pengaturan ini tidak menambah kecepatan.
- Naikkan `OLLAMA_KEEP_ALIVE` agar model tidak dilepas antar pertanyaan; memuat
  ulang menambah ±2,5 detik pada jawaban pertama.

### Jawaban memuat tag `<think>`

Seharusnya sudah tersaring otomatis. Bila masih muncul, versi Ollama Anda
mengabaikan parameter `think`; perbarui dengan `ollama update`.

### Backend CLI "tidak siap"

```powershell
claude --version
codex --version
agy --version
```

Bila salah satu gagal, perkakas itu belum terpasang atau belum berada di `PATH`.
Isikan jalur lengkapnya pada `.env` (`CLAUDE_CLI_PATH`, `CODEX_CLI_PATH`,
`ANTIGRAVITY_CLI_PATH`).

### Backend CLI selesai tanpa jawaban

Umumnya karena sesi CLI tersebut belum masuk akun atau kuotanya habis. Uji
langsung di terminal:

```powershell
"Balas dengan: OK" | claude --print
```

### Jawaban Codex muncul sekaligus, tidak mengalir

Memang demikian. Codex melaporkan jawabannya dalam satu peristiwa
`item.completed`, bukan potongan bertahap. Backend lain mengalir normal.

---

## Antarmuka web

### Halaman kosong atau tanpa gaya

Berkas statis tidak termuat. Pastikan direktori `web\` berisi `index.html`,
`app.js`, dan `styles.css`, lalu muat ulang dengan `Ctrl+F5`.

### Lewat alamat publik, perubahan gaya tidak muncul

Alamat publik (`https://akreditasi.ayolearn.stream`) melewati Cloudflare, yang
mengganti `Cache-Control: no-cache` dari server dengan `max-age=14400` untuk
berkas statis. Peramban lalu memakai `landing.css`, `styles.css`, dan berkas
`.js` simpanannya hingga 4 jam tanpa memeriksa ulang, padahal HTML-nya selalu
baru — halaman tampil dengan susunan baru tetapi gaya lama. Periksa dengan:

```powershell
curl.exe -sI https://akreditasi.ayolearn.stream/static/landing.css
```

Karena itu `landing.html` dan `index.html` memanggil berkas tersebut dengan
`?v=` berisi hash isinya, misalnya `/static/landing.css?v=fafebf62`. Setiap kali
berkas itu diubah, ganti juga nilai `?v=`-nya; nilai baru apa pun berfungsi.
Penyelesaian permanennya ada di dasbor Cloudflare: **Caching → Configuration →
Browser Cache TTL → Respect Existing Headers**.

### Di ponsel, sumber muncul lalu jawaban gagal dengan "network error"

Gejala versi lama: koneksi putus di tengah jeda sebelum jawaban mengalir,
biasanya karena layar terkunci atau aplikasi lain dibuka, dan jawabannya hilang.
Versi sekarang menyambung kembali dengan sendirinya dan menampilkan *"Koneksi
terputus — menyambung kembali"* selama menunggu. Bila pesan lama masih muncul,
peramban memakai `app.js` lama: muat ulang halaman.

Bila yang muncul *"Jawaban ini tidak lagi tersedia di server"*, server dimulai
ulang saat jawaban dibuat; ajukan pertanyaan kembali. Bila *"tidak dapat
disambung kembali"*, jaringan terputus terlalu lama — jawabannya tetap
diselesaikan server dan dapat dibuka dari panel **Percakapan**.

### Di desktop tampilannya seperti ponsel

Tata letak ponsel dipakai bila lebar jendela 960 px atau kurang, dihitung dalam
piksel CSS — setelah skala tampilan Windows dan zoom peramban. Jendela yang
tidak dimaksimalkan pada skala 125% mudah berada di bawah batas itu.
Maksimalkan jendela atau kurangi zoom (`Ctrl+0`). Bila baru saja memperbarui
aplikasi, muat ulang dengan `Ctrl+F5` agar `styles.css` terbaru terpakai.

### Tombol "Buka dokumen" tidak berfungsi

- Setel `ALLOW_OPEN_FILE=1` pada `.env`.
- Fitur ini hanya tersedia di Windows.
- Berkas mungkin sudah tidak ada di sumber; coba **Salin lokasi** lalu periksa
  manual.

### Porta 8026 sudah dipakai

```powershell
.\.venv\Scripts\python.exe main.py serve --port 8426
```

---

## Pengujian

### `ModuleNotFoundError: No module named 'app'`

Jalankan pytest dari akar proyek, bukan dari dalam `tests\`:

```powershell
cd d:\ALFA\akreditasi
.\.venv\Scripts\python.exe -m pytest tests
```

### Pengujian gagal setelah mengubah pengaturan

Suite sengaja mengabaikan `.env` dan membangun `Settings` sendiri di direktori
sementara. Kegagalan setelah perubahan konfigurasi menandakan perubahan pada
**kode**, bukan pada pengaturan.

---

## Memulai dari nol

Bila indeks tampak tidak konsisten dan Anda ingin bersih total:

```powershell
Remove-Item data\index.db*, data\vectors.npz -Force
.\.venv\Scripts\python.exe main.py ingest
```

Cache model di `data\models` tidak perlu dihapus. Dokumen sumber di shared drive
tidak pernah tersentuh oleh langkah mana pun di atas.
