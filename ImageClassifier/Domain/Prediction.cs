
namespace ImageClassifier.Domain;

public class Prediction
{
    public string PredictedLabel { get; set; } = "";

    public float[] Score { get; set; } = Array.Empty<float>();

    public Prediction(string label, float[] score)
    {
        PredictedLabel = label;
        Score = score;
    }
}