using Microsoft.ML.Data;

namespace ImageClassifier.Domain;

public class ImagePrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; } = "";

    public float[] Score { get; set; } = Array.Empty<float>();
}