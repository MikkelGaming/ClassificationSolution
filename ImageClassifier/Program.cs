// File: ./Program.cs
// Target: net9.0
//
// Packages (from your .csproj):
// - Microsoft.ML 4.0.2
// - Microsoft.ML.ImageAnalytics 4.0.2
// - Microsoft.ML.TensorFlow 4.0.2
// - SciSharp.TensorFlow.Redist 2.10.0
// - TensorFlow.NET 0.20.1
//
// Build/Run: dotnet run -c Release

using System;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Image;

namespace MaterialClassifier
{
    internal sealed class Program
    {
        // === Hardcoded inputs (required) ===
        private const string MODEL_DIR = @"./Model/model.savedmodel"; // <-- change to your SavedModel folder
        private const string IMAGE_PATH = @"./can.jpg";         // <-- change to your image path

        // === Optional hardcoded config (set if auto-choices don't match your model) ===
        // Common defaults; adjust if your model uses different node names.
        // Use operation names (without ":0") as ML.NET expects op names.
        private const string MODEL_INPUT_TENSOR = "serving_default_input_1"; // e.g., "serving_default_input_1" or "input_1"
        private const string MODEL_OUTPUT_TENSOR = "StatefulPartitionedCall"; // e.g., "StatefulPartitionedCall" or "Identity"

        // Image preprocessing defaults; adjust if needed.
        private const int IMAGE_WIDTH = 224;
        private const int IMAGE_HEIGHT = 224;
        private const bool CHANNELS_LAST = true; // true for NHWC

        // Class labels order must match your model's output order.
        private static readonly string[] CLASS_LABELS = { "Glass", "Metal", "Plastic" };

        private static int Main(string[] args)
        {
            try
            {
                ValidateHardcodedPaths();

                var ml = new MLContext(seed: 123);

                // Load TF SavedModel.
                var tfModel = ml.Model.LoadTensorFlowModel(MODEL_DIR);

                // (Why) Print schema to help verify tensor names if needed.
                Console.WriteLine("=== TensorFlow Model Schema ===");
                Console.WriteLine(tfModel.GetModelSchema().ToString());

                // Build image → tensor pipeline.
                var imageCol = "image_raw";
                var resizedCol = "image_resized";
                var inputTensorCol = "input_tensor_float";
                var inputOpName = MODEL_INPUT_TENSOR;
                var outputOpName = MODEL_OUTPUT_TENSOR;
                var scoredCol = "tf_scores";

                // Note: LoadImages can take absolute paths in the source column; set imageFolder to "".
                var pipeline =
                    ml.Transforms.LoadImages(outputColumnName: imageCol, imageFolder: "", inputColumnName: nameof(ImageInput.ImagePath))
                    .Append(ml.Transforms.ResizeImages(outputColumnName: resizedCol, imageWidth: IMAGE_WIDTH, imageHeight: IMAGE_HEIGHT, inputColumnName: imageCol, resizing: ImageResizingEstimator.ResizingKind.IsoCrop))
                    .Append(ml.Transforms.ExtractPixels(outputColumnName: inputTensorCol, inputColumnName: resizedCol,
                                                        interleavePixelColors: CHANNELS_LAST, scaleImage: 1f / 255f))
                    // Rename the preprocessed tensor to match TF input op name.
                    .Append(ml.Transforms.CopyColumns(outputColumnName: inputOpName, inputColumnName: inputTensorCol))
                    // Run TF scoring; output column must match TF output op name.
                    .Append(tfModel.ScoreTensorFlowModel(outputColumnNames: new[] { outputOpName },
                                                         inputColumnNames: new[] { inputOpName },
                                                         addBatchDimensionInput: true))
                    // Copy to a stable name for PredictionEngine contract.
                    .Append(ml.Transforms.CopyColumns(outputColumnName: scoredCol, inputColumnName: outputOpName));

                // Fit on a tiny single-row IDataView (no training).
                var data = ml.Data.LoadFromEnumerable(new[] { new ImageInput { ImagePath = IMAGE_PATH } });
                var model = pipeline.Fit(data);

                // Predict.
                var engine = ml.Model.CreatePredictionEngine<ImageInput, TfPrediction>(model);
                var pred = engine.Predict(new ImageInput { ImagePath = IMAGE_PATH });

                if (pred?.Scores is null || pred.Scores.Length == 0)
                    throw new InvalidOperationException("Model returned no scores. Verify output tensor name and shape.");

                // Softmax if logits; safe to apply even if already probabilities (keeps order).
                var probs = Softmax(pred.Scores);

                // Map to labels (best-effort).
                var (bestIdx, bestProb) = ArgMax(probs);
                var bestLabel = bestIdx < CLASS_LABELS.Length ? CLASS_LABELS[bestIdx] : $"class_{bestIdx}";

                Console.WriteLine();
                Console.WriteLine("=== Prediction ===");
                Console.WriteLine($"Image: {IMAGE_PATH}");
                Console.WriteLine($"Predicted: {bestLabel}  (p={bestProb:0.000})");
                Console.WriteLine();

                Console.WriteLine("Top probabilities:");
                var top = probs
                    .Select((p, i) => new { i, p, label = i < CLASS_LABELS.Length ? CLASS_LABELS[i] : $"class_{i}" })
                    .OrderByDescending(x => x.p)
                    .Take(3)
                    .ToList();

                foreach (var t in top)
                    Console.WriteLine($" - {t.label,-8} : {t.p:0.000}");

                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine();
                Console.Error.WriteLine("FATAL: " + ex.Message);
                Console.Error.WriteLine(ex.StackTrace);
                Console.Error.WriteLine();
                Console.Error.WriteLine("Hints:");
                Console.Error.WriteLine(" - Ensure MODEL_DIR points to a TF SavedModel folder (contains saved_model.pb and variables/).");
                Console.Error.WriteLine(" - Confirm MODEL_INPUT_TENSOR and MODEL_OUTPUT_TENSOR match the op names printed in the schema above.");
                Console.Error.WriteLine(" - If your model expects a different size, set IMAGE_WIDTH/IMAGE_HEIGHT accordingly.");
                Console.Error.WriteLine(" - If output class order differs, adjust CLASS_LABELS to match your training.");
                return 1;
            }
        }

        private static void ValidateHardcodedPaths()
        {
            if (!Directory.Exists(MODEL_DIR))
                throw new DirectoryNotFoundException($"MODEL_DIR not found: {MODEL_DIR}");
            if (!File.Exists(IMAGE_PATH))
                throw new FileNotFoundException($"IMAGE_PATH not found: {IMAGE_PATH}");
        }

        // (Why) Numerical stability + predictable top-1 even if logits.
        private static float[] Softmax(float[] logits)
        {
            var max = logits.Max();
            var exps = logits.Select(v => Math.Exp(v - max)).ToArray();
            var sum = exps.Sum();
            if (sum == 0) return Enumerable.Repeat(1f / logits.Length, logits.Length).ToArray();
            return exps.Select(v => (float)(v / sum)).ToArray();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static (int index, float value) ArgMax(float[] values)
        {
            var bestIdx = 0;
            var best = float.MinValue;
            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] > best)
                {
                    best = values[i];
                    bestIdx = i;
                }
            }
            return (bestIdx, best);
        }

        // Input schema for single-row scoring.
        private sealed class ImageInput
        {
            public string ImagePath { get; set; } = string.Empty;
        }

        // Output schema: bind to "tf_scores" vector.
        private sealed class TfPrediction
        {
            [VectorType]
            public float[] Scores { get; set; } = Array.Empty<float>();
        }
    }
}
