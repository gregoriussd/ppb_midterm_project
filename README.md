# Tjarik

Aplikasi ini mengimplementasikan Batik Classification menggunakan model TFLite unquant.

## Dataset:
https://huggingface.co/datasets/muhammadsalmanalfaridzi/Batik-Indonesia  
Dari dataset tersebut, beberapa yang class yang dirasa kurang berkualitas datanya di eliminasi hingga terpilih 28 Class batik, yaitu:  
1. Aceh
2. Barong
3. Merak
4. Betawi
5. Corak Insang
6. Ondel-ondel
7. Megamendung
8. Pring
9. Dayak
10. Gajah
11. Lasem
12. Mataketeran
13. Pala
14. Lumbung
15. Asmat
16. Cendrawasih
17. Tifa
18. Priangan
19. Sekar Jagad
20. Sidoluhur
21. Sidomukti
22. Sogan
23. Lontara
24. Rumah Minang
25. Boraspati
26. Tambal
27. Kawung
28. Parang

## Model:
Model dibuat dengan menggunakan teachable machine. Sebuah platform untuk membuat model AI dengan no-code. Project Teachable Machine dapat di akses melalui link berikut:
https://drive.google.com/file/d/1IOofJecy2H3RwMnJTq-79UIVuICgse6r/view?usp=sharing  
  
Model Teachable machine dapat diexport sebagai TFLite. Model Quantized lebih disarankan untuk aplikasi android karena lebih ringan namun karena error pada konversi maka saya menggunakan Model Floating Point (float32).

## Implementasi Flutter
Implementasi pada flutter dilakukan menggunakan `tflite_flutter` dengan memodifikasi page `camera_preview` dari aplikasi Tjarik versi sebelumnya yang menggunakan Gemini API.  

Perubahan yang dilakukan (berdasarkan alur analisis gambar):

### 1. **`_initializeTfliteInterpreter()`**
Fungsi untuk menginisialisasi interpreter TFLite dari file model yang telah disiapkan. Fungsi ini:
- Memuat model dari local storage menggunakan `_prepareLocalModelPath()`
- Mengekstrak informasi tensor input (dimensi, tipe data)
- Mengekstrak informasi tensor output
- Memuat label dari assets menggunakan `_loadLabels()`
- Handle error dan menyimpan pesan error untuk debugging

### 2. **`_prepareLocalModelPath()`**
Fungsi untuk mempersiapkan path model TFLite secara lokal. Fungsi ini:
- Memeriksa apakah model sudah ada di application support directory
- Jika belum ada, mengekstrak model dari assets
- Menulis model ke local storage dengan `flush: true` untuk memastikan data tersimpan
- Return path lengkap ke file model yang siap digunakan

### 3. **`_loadLabels()`**
Fungsi untuk memuat label dari file `assets/models/labels.txt`. Format file adalah `index label` (contoh: `0 Aceh`). Fungsi ini:
- Parse setiap baris menjadi parts
- Ambil bagian label (index 1 dan seterusnya)
- Return list of String yang berisi nama-nama batik

### 4. **`_centerCropToAspect()`**
Fungsi untuk crop image ke aspect ratio target dengan center positioning. Fungsi ini:
- Menghitung crop dimensions berdasarkan aspect ratio target
- Mengaplikasikan `cropFraction` untuk mengambil hanya bagian center (sesuai UI crop box)
- Menghitung offset untuk center crop
- Return cropped image yang siap di-resize

### 5. **`_preprocessImage()`**
Fungsi untuk preprocessing image sebelum dikirim ke model. Alur preprocessing:
- Decode image dari file ke format pixel
- Center crop menggunakan `_centerCropToAspect()` dengan `_cropBoxFraction` (0.7)
- Resize ke ukuran model input (224x224)
- Normalize pixel values:
  - Untuk `uint8`: konversi ke integer RGB
  - Untuk `float32`: normalize ke range [0.0, 1.0] dengan membagi 255.0
- Return tensor input yang siap untuk inference

### 6. **`_analyzeImageWithTflite()`**
Fungsi utama untuk melakukan inference/analisis gambar menggunakan TFLite. Fungsi ini:
- Menunggu interpreter siap (await `_initializeInterpreterFuture`)
- Preprocessing image dengan `_preprocessImage()`
- Membuat output buffer sesuai output shape model
- Menjalankan inference dengan `_interpreter.run(input, output)`
- Mencari confidence tertinggi dari output predictions
- Return hasil analisis dalam format Map dengan keys: `name`, `origin`, `philosophy`, `confidence`

### 7. **`_buildScaledCameraPreview()`**
Fungsi untuk membangun camera preview yang di-scale untuk mengisi portrait screen tanpa stretch. Fungsi ini:
- Mengambil preview size dari controller
- Menggunakan `FittedBox` dengan `BoxFit.cover` untuk maintain aspect ratio
- Wrap dengan `ClipRect` untuk mencegah overflow
- Rotate preview 90° dengan menukar width-height karena camera native landscape

### 8. **`_buildCropOverlay()`**
Fungsi untuk membangun UI overlay crop box dan gray mask. Fungsi ini:
- Menghitung dimensi crop box berdasarkan `_cropBoxFraction` (0.7) dan target aspect ratio
- Membuat 4 positioned container gray mask untuk area di luar crop box
- Membuat white border box di center untuk visualisasi area yang akan dianalisis
- Return IgnorePointer widget agar overlay tidak mengganggu interaksi camera

## Notes:
- Data origin dan phylosophy belum dibuat dan dimapping untuk di retrieve setelah classification (Future)
- Model masih kurang akurat, suspek saat ini diakibatkan kurangnya hyperparameter tuning dan dataset yang kurang baik