
using System.Numerics;
using ImageClassifier.Domain;
using ImageClassifier.Interface_Adapter;

namespace ImageClassifier.Application;

public class ImageClassifier
{
    IPredictionEngine predictionEngine;
    public ImageClassifier()
    {
        predictionEngine = new PredictionEngine("imageClassifier_model.zip");
    }

    public Prediction ClassifyImage(string ImagePath)
    {
        var p = predictionEngine.Predict(ImagePath);

        var indexed = p.Score
                .Select((s, i) => new { Index = i, Score = s })
                .OrderByDescending(x => x.Score)
                .ToArray();

        return new Prediction(p.PredictedLabel, [indexed[0].Score, indexed[1].Score, indexed[1].Score]);
    }
}