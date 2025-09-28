// Program.cs
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;

public class ImageData
{
    public string ImagePath { get; set; } = "";
    public string Label { get; set; } = "";
}

public class ImagePrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; } = "";

    public float[] Score { get; set; } = Array.Empty<float>();
}

class Program
{
    static void Main()
    {
        try
        {
            // === CONFIG ===
            var baseDir = @"C:\Users\mikke\Pictures\Billedeklassifikation"; // must contain subfolders for each class
            var testImage = @"C:\Projects\C#\ClassificationSolution\ImageClassifier\glass169.jpg"; // example test image (can be absolute)
            var modelPath = "imageClassifier_model.zip";

            var classes = new[] { "Glass", "Metal", "Plastic" };
            var exts = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                { ".jpg", ".jpeg", ".png", ".bmp" };

            // === collect image paths ===
            var trainList = new List<ImageData>();
            var validList = new List<ImageData>();

            Console.WriteLine("=== Scanning folders ===");
            foreach (var cls in classes)
            {
                var trainDir = Path.Combine(baseDir, cls, "train");
                var validDir = Path.Combine(baseDir, cls, "model_test"); // kept your naming

                Console.WriteLine($"\n[CLASS] {cls}");
                Console.WriteLine($" trainDir : {trainDir}  {(Directory.Exists(trainDir) ? "" : "(MISSING)")}");
                Console.WriteLine($" validDir : {validDir}  {(Directory.Exists(validDir) ? "" : "(MISSING)")}");

                int trainCount = 0, validCount = 0;

                if (Directory.Exists(trainDir))
                {
                    foreach (var f in Directory.EnumerateFiles(trainDir, "*.*", SearchOption.AllDirectories))
                    {
                        if (!exts.Contains(Path.GetExtension(f))) continue;
                        trainList.Add(new ImageData { ImagePath = f, Label = cls });
                        trainCount++;
                        Console.WriteLine($"  [train] {cls} -> {f}");
                    }
                }

                if (Directory.Exists(validDir))
                {
                    foreach (var f in Directory.EnumerateFiles(validDir, "*.*", SearchOption.AllDirectories))
                    {
                        if (!exts.Contains(Path.GetExtension(f))) continue;
                        validList.Add(new ImageData { ImagePath = f, Label = cls });
                        validCount++;
                        Console.WriteLine($"  [valid] {cls} -> {f}");
                    }
                }

                Console.WriteLine($"  SUM: train={trainCount}, valid={validCount}");
            }

            Console.WriteLine($"\nTOTALS: train={trainList.Count}, valid={validList.Count}");
            if (trainList.Count == 0 || validList.Count == 0)
            {
                Console.WriteLine("\nNo images found in train/model_test (or extensions do not match).");
                Console.WriteLine("Adjust baseDir or add file extensions to 'exts'.");
                return;
            }

            // === ML setup ===
            var ml = new MLContext(seed: 1);

            // Create IDataViews
            var trainData = ml.Data.LoadFromEnumerable(trainList);
            var validData = ml.Data.LoadFromEnumerable(validList);

            // Preprocessing pipeline: map label to key and load image bytes into ImageBytes column.
            // We'll reuse this transform for both training and validation.
            var preprocessing = ml.Transforms.Conversion.MapValueToKey(
                                    outputColumnName: "LabelAsKey",
                                    inputColumnName: nameof(ImageData.Label))
                                .Append(ml.Transforms.LoadRawImageBytes(
                                    outputColumnName: "ImageBytes",
                                    imageFolder: "",                   // empty -> accepts absolute paths in ImagePath
                                    inputColumnName: nameof(ImageData.ImagePath)));

            // Fit preprocessing on the training data and transform both train and validation sets
            var preprocessingTransformer = preprocessing.Fit(trainData);
            var trainTransformed = preprocessingTransformer.Transform(trainData);
            var validTransformed = preprocessingTransformer.Transform(validData);

            // === Trainer options ===
            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "ImageBytes",
                LabelColumnName = "LabelAsKey",
                ValidationSet = validTransformed,            // <-- this MUST be transformed to include ImageBytes & LabelAsKey
                TestOnTrainSet = false,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
            };

            // Trainer alone (we will append it to preprocessing for final model)
            var trainer = ml.MulticlassClassification.Trainers.ImageClassification(options);

            // Build full pipeline: preprocessing -> trainer -> map predicted key back to value
            var trainingPipeline = preprocessing.Append(trainer)
                                                .Append(ml.Transforms.Conversion.MapKeyToValue(
                                                    outputColumnName: "PredictedLabel",
                                                    inputColumnName: "PredictedLabel"));

            // This will hold the trained or loaded model
            ITransformer model;

            // A list of class labels (will try to get from model metadata if possible)
            string[] labelNames = classes;

            // If saved model exists, load it; otherwise train and save
            if (File.Exists(modelPath))
            {
                Console.WriteLine($"\nModel file found at '{modelPath}'. Loading model...");
                DataViewSchema loadedSchema;
                model = ml.Model.Load(modelPath, out loadedSchema);
                Console.WriteLine("Loaded model successfully.");

                // Try to extract SlotNames metadata for Score column so we can map index -> label
                try
                {
                    var outputSchema = model.GetOutputSchema(trainData.Schema);
                    int scoreColIndex = FindColumnIndex(outputSchema, "Score");
                    if (scoreColIndex >= 0)
                    {
                        var scoreCol = outputSchema[scoreColIndex];
                        VBuffer<ReadOnlyMemory<char>> slotNames = default;
                        try
                        {
                            scoreCol.Annotations.GetValue("SlotNames", ref slotNames);
                            var labels = slotNames.DenseValues().Select(x => x.ToString()).ToArray();
                            if (labels.Length > 0)
                            {
                                labelNames = labels;
                                Console.WriteLine("Extracted label names from model metadata:");
                                for (int i = 0; i < labelNames.Length; i++)
                                    Console.WriteLine($" {i}: {labelNames[i]}");
                            }
                            else
                            {
                                Console.WriteLine("SlotNames metadata present but empty. Falling back to provided classes array.");
                            }
                        }
                        catch
                        {
                            Console.WriteLine("Could not read SlotNames metadata via GetValue. Falling back to provided classes array.");
                        }
                    }
                    else
                    {
                        Console.WriteLine("Score column not found in model output schema. Cannot extract class names from metadata.");
                    }
                }
                catch (Exception exMeta)
                {
                    Console.WriteLine("Failed to extract label metadata: " + exMeta.Message);
                }
            }
            else
            {
                Console.WriteLine("\nNo saved model found. Training a new model...");
                Console.WriteLine("\n=== Training model ===");

                model = trainingPipeline.Fit(trainData);

                Console.WriteLine("\nSaving model to " + modelPath);
                ml.Model.Save(model, trainData.Schema, modelPath);
                Console.WriteLine("Model saved.");

                // After training, try to get slot names from model schema as above
                try
                {
                    var outputSchema = model.GetOutputSchema(trainData.Schema);
                    int scoreColIndex = FindColumnIndex(outputSchema, "Score");
                    if (scoreColIndex >= 0)
                    {
                        var scoreCol = outputSchema[scoreColIndex];
                        VBuffer<ReadOnlyMemory<char>> slotNames = default;
                        try
                        {
                            scoreCol.Annotations.GetValue("SlotNames", ref slotNames);
                            var labels = slotNames.DenseValues().Select(x => x.ToString()).ToArray();
                            if (labels.Length > 0)
                            {
                                labelNames = labels;
                                Console.WriteLine("Extracted label names from trained model metadata:");
                                for (int i = 0; i < labelNames.Length; i++)
                                    Console.WriteLine($" {i}: {labelNames[i]}");
                            }
                            else
                            {
                                Console.WriteLine("SlotNames metadata present but empty in trained model. Using provided classes array.");
                            }
                        }
                        catch
                        {
                            Console.WriteLine("Could not read SlotNames metadata from trained model. Using provided classes array.");
                        }
                    }
                }
                catch (Exception exMeta)
                {
                    Console.WriteLine("Failed to extract label metadata from trained model: " + exMeta.Message);
                }
            }

            // === Create prediction engine (model includes preprocessing) ===
            var engine = ml.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

            // Single test image prediction (call without passing explicit null)
            if (File.Exists(testImage))
            {
                var pred = engine.Predict(new ImageData { ImagePath = testImage });
                PrintPrediction(testImage, pred, labelNames: labelNames);
            }
            else
            {
                Console.WriteLine($"\nTest image not found: {testImage}. Skipping single-image prediction.");
            }

            // Print predictions for a small subset of validation images to sanity-check
            Console.WriteLine("\n=== Quick validation sample predictions ===");
            foreach (var sample in validList.Take(10))
            {
                var p = engine.Predict(sample);
                PrintPrediction(sample.ImagePath, p, sample.Label, labelNames);
            }

            Console.WriteLine("\nDone.");
        }
        catch (Exception ex)
        {
            Console.WriteLine("\n❌ Exception:");
            Console.WriteLine(ex.ToString());
        }

        Console.WriteLine("\nPress any key to exit...");
        Console.ReadKey();
    }

    /// <summary>
    /// Find a column index by name in a DataViewSchema (compatibility helper for ML.NET versions lacking TryGetColumnIndex).
    /// </summary>
    static int FindColumnIndex(DataViewSchema schema, string columnName)
    {
        for (int i = 0; i < schema.Count; i++)
            if (schema[i].Name.Equals(columnName, StringComparison.Ordinal))
                return i;
        return -1;
    }

    // Allow trueLabel to be nullable and labelNames to be nullable as well
    static void PrintPrediction(string imagePath, ImagePrediction pred, string? trueLabel = null, string[]? labelNames = null)
    {
        Console.WriteLine($"\nImage: {imagePath}");
        if (!string.IsNullOrEmpty(trueLabel))
            Console.WriteLine($" True label: {trueLabel}");

        Console.WriteLine($" Predicted label: {pred?.PredictedLabel ?? "<null>"}");

        if (pred?.Score != null && pred.Score.Length > 0)
        {
            // Print top score (and optionally top-3)
            var indexed = pred.Score
                .Select((s, i) => new { Index = i, Score = s })
                .OrderByDescending(x => x.Score)
                .ToArray();

            var best = indexed.First();
            string bestLabel = labelNames != null && best.Index < labelNames.Length ? labelNames[best.Index] : $"index {best.Index}";
            Console.WriteLine($" Confidence (best): {best.Score:P2} -> {bestLabel}");

            // optionally show top-3 scores with labels
            Console.WriteLine(" Top scores:");
            foreach (var item in indexed.Take(3))
            {
                string lbl = labelNames != null && item.Index < labelNames.Length ? labelNames[item.Index] : $"idx {item.Index}";
                Console.WriteLine($"  - {lbl} : {item.Score:P2} (idx={item.Index})");
            }
        }
    }
}
