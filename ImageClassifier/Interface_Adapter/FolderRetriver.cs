
using ImageClassifier.Domain;

namespace ImageClassifier.Application;

public static class FolderRetriver
{
    private static HashSet<string> exts = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".jpg", ".jpeg", ".png", ".bmp" };

    public static void ScanFolders(string[] classes, ref List<ImageData> trainList, ref List<ImageData> validList, string baseDir)
    {
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
    }
}