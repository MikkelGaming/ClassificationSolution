
namespace ImageClassifier;

class Program
{
    static void Main()
    {
        try
        {
            PredictionEngine predictionEngine = new("imageClassifier_model.zip");


            string testImage = @"C:\Projects\C#\ClassificationSolution\ImageClassifier\glass169.jpg"; // example test image (can be absolute)

            string[] labelNames = { "Glass", "Metal", "Plastic" };

            // Single test image prediction (call without passing explicit null)
            if (File.Exists(testImage))
            {
                var pred = predictionEngine.Predict(testImage);
                PrintPrediction(testImage, pred, labelNames: labelNames);
            }
            else
            {
                Console.WriteLine($"\nTest image not found: {testImage}. Skipping single-image prediction.");
            }

            // Print predictions for a small subset of validation images to sanity-check
            Console.WriteLine("\n=== Quick validation sample predictions ===");
            foreach (var sample in ModelTrainer.GetValidList().Take(10))
            {
                var p = predictionEngine.Predict(sample);
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
