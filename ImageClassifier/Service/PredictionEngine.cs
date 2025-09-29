
using ImageClassifier.Domain;
using ImageClassifier.Interface_Adapter;
using Microsoft.ML;

namespace ImageClassifier.Service;

public class PredictionEngine : IPredictionEngine
{
    private PredictionEngine<ImageData, ImagePrediction> engine;
    public PredictionEngine(string modelPath)
    {
        MLContext mLContext = new(seed: 1);

        ITransformer model = ModelTrainer.GetModel(modelPath, mLContext);

        // === Create prediction engine (model includes preprocessing) ===
        engine = mLContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
    }

    /// <summary>
    /// Predicts class of image.
    /// </summary>
    /// <param name="path">Path of the image</param>
    /// <returns></returns>
    public ImagePrediction Predict(string path)
    {
        return engine.Predict(new ImageData { ImagePath = path });
    }

    /// <summary>
    /// Predicts class of image.
    /// </summary>
    /// <param name="path">Instance of ImageData</param>
    /// <returns></returns>
    public ImagePrediction Predict(ImageData imageData)
    {
        return engine.Predict(imageData);
    }
}