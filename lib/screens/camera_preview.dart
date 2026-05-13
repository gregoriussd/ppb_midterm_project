import 'dart:io';
import 'dart:math';

import 'package:firebase_ai/firebase_ai.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_image_compress/flutter_image_compress.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:tjarik/models/batik_model.dart';
import 'package:tjarik/services/database_service.dart';
import 'package:tjarik/services/notification_service.dart';
import 'package:tjarik/widgets/bottom_navbar.dart';

class CameraPreviewScreen extends StatefulWidget {
  final CameraDescription camera;
  final FirebaseStorage storage;
  final GenerativeModel model;

  const CameraPreviewScreen({
    super.key,
    required this.camera,
    required this.storage,
    required this.model,
  });

  @override
  State<CameraPreviewScreen> createState() => _CameraPreviewScreenState();
}

class _CameraPreviewScreenState extends State<CameraPreviewScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  late Future<void> _initializeInterpreterFuture;

  final ImagePicker _picker = ImagePicker();
  Interpreter? _interpreter;
  List<String> _labels = [];
  String? _interpreterInitError;
  bool _isUploading = false;
  int _currentNavIndex = 2;

  int _inputHeight = 224;
  int _inputWidth = 224;
  int _inputChannels = 3;
  int _outputLength = 0;
  List<int> _outputShape = [];
  TensorType _inputType = TensorType.float32;
  static const double _modelInputSize = 224.0;
  static const double _confidenceThreshold = 0.3;
  static const double _cropBoxFraction = 0.7;
  static const double _cropBoxBorderRadius = 12.0;
  static const double _cropBoxBorderWidth = 2.0;
  static const Color _cropMaskColor = Color(0x99000000);

  @override
  void initState() {
    super.initState();
    _controller = CameraController(widget.camera, ResolutionPreset.high);
    _initializeControllerFuture = _controller.initialize();
    _initializeInterpreterFuture = _initializeTfliteInterpreter();
  }

  @override
  void dispose() {
    _interpreter?.close();
    _controller.dispose();
    super.dispose();
  }

  Future<void> _initializeTfliteInterpreter() async {
    try {
      final modelPath = await _prepareLocalModelPath();
      _interpreter = Interpreter.fromFile(File(modelPath));
      final inputTensor = _interpreter!.getInputTensor(0);
      final inputShape = inputTensor.shape;
      _inputType = inputTensor.type;
      if (inputShape.length == 4) {
        _inputHeight = inputShape[1];
        _inputWidth = inputShape[2];
        _inputChannels = inputShape[3];
      } else if (inputShape.length == 3) {
        _inputHeight = inputShape[0];
        _inputWidth = inputShape[1];
        _inputChannels = inputShape[2];
      }

      final outputTensor = _interpreter!.getOutputTensor(0);
      final outputShape = outputTensor.shape;
      _outputShape = outputShape;
      _outputLength = outputShape.isNotEmpty ? outputShape.last : 0;

      _labels = await _loadLabels();
      if (_outputLength == 0) {
        _outputLength = _labels.length;
      }
      _interpreterInitError = null;
    } catch (e) {
      _interpreterInitError = 'Gagal inisialisasi TFLite: $e';
      _interpreter?.close();
      _interpreter = null;
    }
  }

  Future<String> _prepareLocalModelPath() async {
    final supportDir = await getApplicationSupportDirectory();
    final modelFile = File('${supportDir.path}/model_unquant.tflite');

    if (!await modelFile.exists()) {
      ByteData? modelBytes;
      try {
        final loaded = await rootBundle.load(
          'assets/models/model_unquant.tflite',
        );
        if (loaded.lengthInBytes > 0) {
          modelBytes = loaded;
        }
      } catch (_) {}

      if (modelBytes == null) {
        throw Exception(
          'Model TFLite tidak ditemukan di assets. Periksa pubspec.yaml dan path model.',
        );
      }

      await modelFile.writeAsBytes(
        modelBytes.buffer.asUint8List(),
        flush: true,
      );
    }

    return modelFile.path;
  }

  Future<List<String>> _loadLabels() async {
    final labelsFile = await rootBundle.loadString('assets/models/labels.txt');
    final lines = labelsFile.trim().split('\n');
    final labels = <String>[];

    for (final line in lines) {
      final parts = line.trim().split(' ');
      if (parts.length >= 2) {
        labels.add(parts.sublist(1).join(' '));
      }
    }

    return labels;
  }

  img.Image _centerCropToAspect(
    img.Image source,
    double targetAspect,
    double cropFraction,
  ) {
    final srcWidth = source.width;
    final srcHeight = source.height;
    final srcAspect = srcWidth / srcHeight;

    int cropWidth = srcWidth;
    int cropHeight = srcHeight;

    if (srcAspect > targetAspect) {
      cropWidth = (srcHeight * targetAspect).round();
    } else if (srcAspect < targetAspect) {
      cropHeight = (srcWidth / targetAspect).round();
    }

    if (cropFraction > 0 && cropFraction < 1) {
      cropWidth = max(1, (cropWidth * cropFraction).round());
      cropHeight = max(1, (cropHeight * cropFraction).round());
    }

    final offsetX = max(0, ((srcWidth - cropWidth) / 2).round());
    final offsetY = max(0, ((srcHeight - cropHeight) / 2).round());

    return img.copyCrop(
      source,
      x: offsetX,
      y: offsetY,
      width: cropWidth,
      height: cropHeight,
    );
  }

  Future<Object> _preprocessImage(File imageFile) async {
    final bytes = await imageFile.readAsBytes();
    final decoded = img.decodeImage(bytes);

    if (decoded == null) {
      throw Exception('Gagal decode image');
    }

    if (_inputChannels != 3) {
      throw Exception('Channel input model tidak didukung: $_inputChannels');
    }

    final targetWidth = _inputWidth > 0 ? _inputWidth : _modelInputSize.toInt();
    final targetHeight = _inputHeight > 0
        ? _inputHeight
        : _modelInputSize.toInt();
    final targetAspect = targetWidth / targetHeight;
    final cropped = _centerCropToAspect(
      decoded,
      targetAspect,
      _cropBoxFraction,
    );
    final resized = img.copyResize(
      cropped,
      width: targetWidth,
      height: targetHeight,
    );

    if (_inputType == TensorType.uint8) {
      final input = List.generate(
        1,
        (_) => List.generate(
          _inputHeight,
          (y) => List.generate(_inputWidth, (x) {
            final pixel = resized.getPixelSafe(x, y);
            return <int>[pixel.r.toInt(), pixel.g.toInt(), pixel.b.toInt()];
          }),
        ),
      );
      return input;
    }

    final input = List.generate(
      1,
      (_) => List.generate(
        _inputHeight,
        (y) => List.generate(_inputWidth, (x) {
          final pixel = resized.getPixelSafe(x, y);
          return <double>[
            pixel.r.toDouble() / 255.0,
            pixel.g.toDouble() / 255.0,
            pixel.b.toDouble() / 255.0,
          ];
        }),
      ),
    );

    return input;
  }

  Future<Map<String, dynamic>> _analyzeImageWithTflite(XFile image) async {
    await _initializeInterpreterFuture;

    if (_interpreter == null) {
      throw Exception(
        _interpreterInitError ?? 'TFLite interpreter tidak tersedia.',
      );
    }

    if (_labels.isEmpty) {
      throw Exception('Labels tidak berhasil dimuat.');
    }

    try {
      final imageFile = File(image.path);
      final input = await _preprocessImage(imageFile);

      final outputLength = _outputLength > 0 ? _outputLength : _labels.length;
      Object output;
      if (_outputShape.length == 2 && _outputShape[0] == 1) {
        output = List<List<double>>.generate(
          1,
          (_) => List<double>.filled(outputLength, 0.0),
        );
      } else {
        output = List<double>.filled(outputLength, 0.0);
      }

      _interpreter!.run(input, output);

      final predictions = output is List<List<double>>
          ? output.first
          : output as List<double>;

      double maxConfidence = 0.0;
      int maxIndex = -1;

      final compareLength = min(predictions.length, _labels.length);
      for (int i = 0; i < compareLength; i++) {
        final confidence = predictions[i];
        if (confidence > maxConfidence) {
          maxConfidence = confidence;
          maxIndex = i;
        }
      }

      if (maxConfidence < _confidenceThreshold ||
          maxIndex < 0 ||
          maxIndex >= _labels.length) {
        return {
          'name': 'Tidak ditemukan',
          'origin': '-',
          'philosophy': '-',
          'confidence': 0.0,
        };
      }

      final detectedName = _labels[maxIndex];

      return {
        'name': detectedName,
        'origin': '-',
        'philosophy': '-',
        'confidence': maxConfidence.clamp(0.0, 1.0),
      };
    } catch (e) {
      throw Exception('Inference error: $e');
    }
  }

  // Future<Map<String, dynamic>> _analyzeImage(XFile image) async {
  //   const prompt = """
  //     Analisis gambar ini dan identifikasi apakah ini motif batik.
  //     Balas HANYA JSON valid, tanpa teks tambahan, dengan format:
  //     {"name":"...", "origin":"...", "philosophy":"...", "confidence":0.0}

  //     Aturan:
  //     - "name": nama motif batik, atau "Tidak ditemukan" jika bukan batik.
  //     - "origin": daerah asal motif (contoh: "Yogyakarta"). Jika tidak ditemukan, isi "-".
  //     - "philosophy": filosofi singkat motif. Jika tidak ditemukan, isi "-".
  //     - "confidence": angka desimal 0.0 sampai 1.0.

  //     Jika bukan motif batik, balas:
  //     {"name":"Tidak ditemukan", "origin":"-", "philosophy":"-", "confidence":0.0}
  //     """;

  //   final bytes = await image.readAsBytes();
  //   final mimeType = image.mimeType ?? 'image/jpeg';

  //   final response = await widget.model.generateContent([
  //     Content.text(prompt),
  //     Content.inlineData(mimeType, bytes),
  //   ]);

  //   final raw = (response.text ?? '{}').trim();
  //   final cleanedRaw = raw
  //       .replaceAll('```json', '')
  //       .replaceAll('```', '')
  //       .trim();

  //   try {
  //     final data = jsonDecode(cleanedRaw) as Map<String, dynamic>;
  //     final parsedConfidence =
  //         double.tryParse((data['confidence'] ?? '0').toString()) ?? 0.0;

  //     return {
  //       'name': (data['name'] ?? 'Tidak ditemukan').toString(),
  //       'origin': (data['origin'] ?? '-').toString(),
  //       'philosophy': (data['philosophy'] ?? '-').toString(),
  //       'confidence': parsedConfidence.clamp(0.0, 1.0),
  //     };
  //   } catch (_) {
  //     return {
  //       'name': 'Tidak ditemukan',
  //       'origin': '-',
  //       'philosophy': '-',
  //       'confidence': 0.0,
  //     };
  //   }
  // }

  Future<File> _compressTo500Square(XFile image) async {
    final bytes = await image.readAsBytes();
    final decoded = img.decodeImage(bytes);

    if (decoded == null) {
      return File(image.path);
    }

    final square = img.copyResizeCropSquare(decoded, size: 500);
    final squareBytes = Uint8List.fromList(img.encodeJpg(square, quality: 85));

    final compressedBytes = await FlutterImageCompress.compressWithList(
      squareBytes,
      quality: 80,
      format: CompressFormat.jpeg,
      minWidth: 500,
      minHeight: 500,
      keepExif: false,
    );

    final tempDir = await getTemporaryDirectory();
    final outFile = File(
      '${tempDir.path}/${DateTime.now().millisecondsSinceEpoch}_500.jpg',
    );
    await outFile.writeAsBytes(compressedBytes, flush: true);
    return outFile;
  }

  Future<String> _uploadImage(File imageFile) async {
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final imageName = '${timestamp}_compressed.jpg';
    final imagePath = 'user/$imageName';
    final ref = widget.storage.ref().child(imagePath);

    await ref.putFile(imageFile);
    final url = ref.getDownloadURL();
    return url;
  }

  Future<void> _showAnalysisAndAskSave({
    required XFile image,
    required Map<String, dynamic> analysis,
  }) async {
    if (!mounted) return;

    final motifName = (analysis['name'] ?? 'Tidak ditemukan').toString();
    final origin = (analysis['origin'] ?? '-').toString();
    final philosophy = (analysis['philosophy'] ?? '-').toString();
    final confidence =
        double.tryParse((analysis['confidence'] ?? '0').toString()) ?? 0.0;

    String personalNote = '';

    final shouldSave = await showDialog<bool>(
      context: context,
      builder: (context) {
        final screenWidth = MediaQuery.of(context).size.width;
        final dialogWidth = screenWidth > 480 ? 420.0 : screenWidth * 0.9;

        return AlertDialog(
          title: const Text('Hasil Analisis'),
          content: SizedBox(
            width: dialogWidth,
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: AspectRatio(
                      aspectRatio: 4 / 3,
                      child: Image.file(File(image.path), fit: BoxFit.cover),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text('Nama motif: ${motifName.isNotEmpty ? motifName : '-'}'),
                  const SizedBox(height: 8),
                  Text('Asal: ${origin.isNotEmpty ? origin : '-'}'),
                  const SizedBox(height: 8),
                  Text('Filosofi: ${philosophy.isNotEmpty ? philosophy : '-'}'),
                  const SizedBox(height: 8),
                  Text('Confidence: ${(confidence * 100).toStringAsFixed(1)}%'),
                  const SizedBox(height: 12),
                  TextField(
                    onChanged: (value) {
                      personalNote = value;
                    },
                    maxLines: 3,
                    decoration: const InputDecoration(
                      labelText: 'Catatan pribadi',
                      hintText: 'Tulis catatan kamu di sini...',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ],
              ),
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Batal'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Simpan'),
            ),
          ],
        );
      },
    );

    if (shouldSave == true) {
      final user = FirebaseAuth.instance.currentUser;
      if (user != null) {
        final compressedFile = await _compressTo500Square(image);

        BatikScan newScan = BatikScan(
          userId: user.uid,
          motifName: motifName,
          origin: origin,
          philosophy: philosophy,
          imageUrl: await _uploadImage(compressedFile),
          confidence: confidence,
          personalNote: personalNote,
          createdAt: DateTime.now(),
        );
        await DatabaseService().saveBatikScan(newScan);
        await NotificationService.createNotification(
          id: DateTime.now().millisecondsSinceEpoch.remainder(100000),
          title: 'Tersimpan',
          body: 'Hasil scan "$motifName" berhasil disimpan ke Firebase.',
          payload: {'scanAt': newScan.createdAt.toIso8601String()},
        );
      }

      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Tersimpan ke Firebase')));
    }
  }

  Future<void> takePictureWithLocalTflite() async {
    if (_isUploading) return;

    setState(() {
      _isUploading = true;
    });

    try {
      await _initializeControllerFuture;
      final image = await _controller.takePicture();

      final analysis = await _analyzeImageWithTflite(image);
      await NotificationService.createNotification(
        id: DateTime.now().millisecondsSinceEpoch.remainder(100000),
        title: 'Analisis selesai',
        body:
            '${(analysis['name'] ?? 'Tidak ditemukan')} — Confidence: ${((analysis['confidence'] ?? 0.0) * 100).toStringAsFixed(1)}%',
      );
      await _showAnalysisAndAskSave(image: image, analysis: analysis);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Gagal proses foto: $e')));
      }
      return;
    } finally {
      if (mounted) {
        setState(() {
          _isUploading = false;
        });
      }
    }
  }

  Future<void> pickFromGallery() async {
    if (_isUploading) return;

    try {
      final image = await _picker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 90,
      );

      setState(() => _isUploading = true);
      if (image == null) return;

      final analysis = await _analyzeImageWithTflite(image);
      await NotificationService.createNotification(
        id: DateTime.now().millisecondsSinceEpoch.remainder(100000),
        title: 'Analisis selesai',
        body:
            '${(analysis['name'] ?? 'Tidak ditemukan')} — Confidence: ${((analysis['confidence'] ?? 0.0) * 100).toStringAsFixed(1)}%',
      );
      await _showAnalysisAndAskSave(image: image, analysis: analysis);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Gallery upload failed: $e')));
      }
      return;
    } finally {
      if (mounted) setState(() => _isUploading = false);
    }
  }

  void navigateDashboard() {
    if (!context.mounted) return;
    Navigator.pushReplacementNamed(context, 'home');
  }

  Widget _buildScaledCameraPreview(Size size) {
    final previewSize = _controller.value.previewSize;
    if (previewSize == null) {
      return CameraPreview(_controller);
    }

    return ClipRect(
      child: SizedBox(
        width: size.width,
        height: size.height,
        child: FittedBox(
          fit: BoxFit.cover,
          child: SizedBox(
            width: previewSize.height,
            height: previewSize.width,
            child: CameraPreview(_controller),
          ),
        ),
      ),
    );
  }

  Widget _buildCropOverlay(Size size) {
    final targetAspect = _inputWidth > 0 && _inputHeight > 0
        ? _inputWidth / _inputHeight
        : 1.0;
    final maxBoxWidth = size.width * _cropBoxFraction;
    final maxBoxHeight = size.height * _cropBoxFraction;

    double boxWidth = maxBoxWidth;
    double boxHeight = boxWidth / targetAspect;

    if (boxHeight > maxBoxHeight) {
      boxHeight = maxBoxHeight;
      boxWidth = boxHeight * targetAspect;
    }

    final left = (size.width - boxWidth) / 2;
    final top = (size.height - boxHeight) / 2;

    return IgnorePointer(
      child: Stack(
        children: [
          Positioned(
            left: 0,
            right: 0,
            top: 0,
            height: top,
            child: Container(color: _cropMaskColor),
          ),
          Positioned(
            left: 0,
            top: top,
            width: left,
            height: boxHeight,
            child: Container(color: _cropMaskColor),
          ),
          Positioned(
            right: 0,
            top: top,
            width: left,
            height: boxHeight,
            child: Container(color: _cropMaskColor),
          ),
          Positioned(
            left: 0,
            right: 0,
            top: top + boxHeight,
            bottom: 0,
            child: Container(color: _cropMaskColor),
          ),
          Positioned(
            left: left,
            top: top,
            width: boxWidth,
            height: boxHeight,
            child: Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(_cropBoxBorderRadius),
                border: Border.all(
                  color: Colors.white,
                  width: _cropBoxBorderWidth,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "Scan Your Batik",
          style: TextStyle(
            fontSize: 28,
            fontWeight: FontWeight.bold,
            color: Colors.black,
          ),
        ),
        centerTitle: false,
        elevation: 0,
        backgroundColor: const Color(0xFFF7F8FA),
      ),
      bottomNavigationBar: BottomNavBar(
        currentIndex: _currentNavIndex,
        onTap: (index) {
          setState(() => _currentNavIndex = index);
        },
      ),
      body: Stack(
        children: [
          FutureBuilder<void>(
            future: _initializeControllerFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState != ConnectionState.done ||
                  !_controller.value.isInitialized) {
                return const Center(child: CircularProgressIndicator());
              }

              return LayoutBuilder(
                builder: (context, constraints) {
                  final size = constraints.biggest;
                  return Stack(
                    fit: StackFit.expand,
                    children: [
                      _buildScaledCameraPreview(size),
                      _buildCropOverlay(size),
                    ],
                  );
                },
              );
            },
          ),
          if (_isUploading)
            Positioned.fill(
              child: AbsorbPointer(
                absorbing: true,
                child: Container(
                  color: Colors.black45,
                  child: const Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        CircularProgressIndicator(),
                        SizedBox(height: 12),
                        Text(
                          'Uploading...',
                          style: TextStyle(color: Colors.white),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      floatingActionButton: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          FloatingActionButton(
            heroTag: 'cameraGalleryBtn',
            onPressed: _isUploading ? null : pickFromGallery,
            child: const Icon(Icons.photo_library),
          ),
          const SizedBox(width: 100),
          FloatingActionButton(
            heroTag: 'cameraShutterBtn',
            onPressed: _isUploading ? null : takePictureWithLocalTflite,
            child: const Icon(Icons.camera_alt),
          ),
          const SizedBox(width: 100),
          FloatingActionButton(
            heroTag: 'cameraSwitchBtn',
            onPressed: () {
              // FOR FUTURE FEATURE
            },
            child: Icon(Icons.cameraswitch),
          ),
        ],
      ),
    );
  }
}
