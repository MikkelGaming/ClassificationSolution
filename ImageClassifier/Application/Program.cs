
using ImageClassifier.Domain;
using ImageClassifier.Interface_Adapter;

namespace ImageClassifier.Application;

class Program
{
    static void Main()
    {
        try
        {
            IPredictionEngine predictionEngine = new PredictionEngine("imageClassifier_model.zip");

            string[] labelNames = { "Glass", "Metal", "Plastic" };
            List<ImageData> tl = [];
            List<ImageData> vl = [];
            FolderRetriver.ScanFolders(labelNames, ref tl, ref vl, @"../Images");

            // Print predictions for a small subset of validation images to sanity-check
            Console.WriteLine("\n=== Quick validation sample predictions ===");
            foreach (var sample in vl.Take(10))
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

    protected static void PrintPrediction(string imagePath, ImagePrediction pred, string? trueLabel = null, string[]? labelNames = null)
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
