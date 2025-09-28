using System;
using System.Collections.Generic;
using System.IO;
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
    [ColumnName("PredictedLabel")] public string PredictedLabel { get; set; } = "";
    public float[] Score { get; set; } = Array.Empty<float>();
}

class Program
{
    static void Main()
    {
        try
        {
            var baseDir = @"C:\Users\mikke\Pictures\Billedeklassifikation";           // <- skal indeholde \bears, \cats, \dogs
            var testImage = @"C:\Projects\C#\ClassificationSolution\ImageClassifier\glass169.jpg";  // et vilkårligt testbillede fra unknown

            var ml = new MLContext(seed: 1);

            var classes = new[] { "Glass", "Metal", "Plastic" };
            var exts = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                { ".jpg", ".jpeg", ".png", ".bmp" /* tilføj fx ".jfif", ".webp" hvis du bruger dem */ };

            var trainList = new List<ImageData>();
            var validList = new List<ImageData>();

            Console.WriteLine("=== Scanner mapper ===");
            foreach (var cls in classes)
            {
                var trainDir = Path.Combine(baseDir, cls, "train");
                var validDir = Path.Combine(baseDir, cls, "model_test");

                Console.WriteLine($"\n[CLASS] {cls}");
                Console.WriteLine($" trainDir : {trainDir}  {(Directory.Exists(trainDir) ? "" : "(FINDES IKKE!)")}");
                Console.WriteLine($" validDir : {validDir}  {(Directory.Exists(validDir) ? "" : "(FINDES IKKE!)")}");

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
                Console.WriteLine("\nIngen billeder fundet i train/model_test (eller filendelser matcher ikke).");
                Console.WriteLine("Tilføj evt. flere endelser i 'exts' (fx .jfif, .webp) eller ret baseDir.");

                return;
            }

            // --- ML pipeline ---
            var trainData = ml.Data.LoadFromEnumerable(trainList);
            var validData = ml.Data.LoadFromEnumerable(validList);

            var pipeline =
                ml.Transforms.Conversion.MapValueToKey("LabelAsKey", nameof(ImageData.Label))
                .Append(ml.Transforms.LoadRawImageBytes(
                    outputColumnName: "ImageBytes",          // This will be byte[] (VarVector<Byte>)
                    imageFolder: "",                          // Empty for absolute paths
                    inputColumnName: nameof(ImageData.ImagePath)))
                .Append(ml.MulticlassClassification.Trainers.ImageClassification(new ImageClassificationTrainer.Options
                {
                    FeatureColumnName = "ImageBytes",         // ✅ VarVector<Byte>
                    LabelColumnName = "LabelAsKey",
                    Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                    ValidationSet = validData,
                    MetricsCallback = (metrics) => Console.WriteLine(metrics),
                    TestOnTrainSet = false
                }))
                .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            Console.WriteLine("\nTræner modellen...");
            var model = pipeline.Fit(trainData);

            var engine = ml.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var pred = engine.Predict(new ImageData { ImagePath = testImage });

            // find bedste score (kun til visning)
            int bestIdx = -1; float bestScore = float.NegativeInfinity;
            if (pred.Score != null && pred.Score.Length > 0)
            {
                bestIdx = 0; bestScore = pred.Score[0];
                for (int i = 1; i < pred.Score.Length; i++)
                    if (pred.Score[i] > bestScore) { bestIdx = i; bestScore = pred.Score[i]; }
            }

            ml.Model.Save(model, trainData.Schema, "model.zip");

            Console.WriteLine("\n=== Klassifikation (ren ML.NET) ===");
            Console.WriteLine($"Billede: {testImage}");
            Console.WriteLine($"Label:   {pred.PredictedLabel}");
            Console.WriteLine($"Score:   {bestScore:P2}");
        }
        catch (Exception ex)
        {
            Console.WriteLine("\n❌ Exception:");
            Console.WriteLine(ex.ToString());
        }

        Console.Read();
    }
}


