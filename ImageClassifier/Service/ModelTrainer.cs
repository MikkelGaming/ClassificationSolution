using ImageClassifier.Application;
using ImageClassifier.Domain;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;

namespace ImageClassifier.Service;

public static class ModelTrainer
{
    private static string baseDir = @"../Images"; // must contain subfolders for each class
    private static string[] classes = new[] { "Glass", "Metal", "Plastic" };
    private static List<ImageData> trainList = new();
    private static List<ImageData> validList = new();
    private static string _modelPath = "";
    private static MLContext? _mL;


    public static ITransformer GetModel(string modelPath, MLContext mL)
    {
        _modelPath = modelPath;
        _mL = mL;

        FolderRetriver.ScanFolders(classes, ref trainList, ref validList, baseDir);

        // If saved model exists, load it; otherwise train and save
        if (File.Exists(modelPath))
        {
            return LoadModel();
        }
        else
        {
            return TrainModel();
        }
    }

    private static ITransformer LoadModel()
    {
        Console.WriteLine($"\nModel file found at '{_modelPath}'. Loading model...");
        DataViewSchema loadedSchema;
        ITransformer model = _mL.Model.Load(_modelPath, out loadedSchema);
        Console.WriteLine("Loaded model successfully.");

        string[] labelNames = classes;
        var trainData = _mL.Data.LoadFromEnumerable(trainList);

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
        return model;
    }

    private static ITransformer TrainModel()
    {
        Console.WriteLine("\nNo saved model found. Training a new model...");

        Console.WriteLine("Preparing training data...");
        // Create IDataViews
        var trainData = _mL.Data.LoadFromEnumerable(trainList);
        var validData = _mL.Data.LoadFromEnumerable(validList);

        // Preprocessing pipeline: map label to key and load image bytes into ImageBytes column.
        // We'll reuse this transform for both training and validation.
        var preprocessing = _mL.Transforms.Conversion.MapValueToKey(
                                outputColumnName: "LabelAsKey",
                                inputColumnName: nameof(ImageData.Label))
                            .Append(_mL.Transforms.LoadRawImageBytes(
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
        var trainer = _mL.MulticlassClassification.Trainers.ImageClassification(options);

        // Build full pipeline: preprocessing -> trainer -> map predicted key back to value
        var trainingPipeline = preprocessing.Append(trainer)
                                            .Append(_mL.Transforms.Conversion.MapKeyToValue(
                                                outputColumnName: "PredictedLabel",
                                                inputColumnName: "PredictedLabel"));

        // This will hold the trained or loaded model
        ITransformer model;

        // A list of class labels (will try to get from model metadata if possible)
        string[] labelNames = classes;

        Console.WriteLine("\n=== Training model ===");
        model = trainingPipeline.Fit(trainData);

        Console.WriteLine("\nSaving model to " + _modelPath);
        _mL.Model.Save(model, trainData.Schema, _modelPath);
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

        return model;
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


    public static List<ImageData> GetValidList()
    {
        return [.. validList];
    }
}