
using ImageClassifier.Domain;

namespace ImageClassifier.Interface_Adapter;

public interface IPredictionEngine
{
    /// <summary>
    /// Predicts class of image.
    /// </summary>
    /// <param name="path">Path of the image</param>
    /// <returns></returns>
    ImagePrediction Predict(string path);

    /// <summary>
    /// Predicts class of image.
    /// </summary>
    /// <param name="imageData">Instance of ImageData</param>
    /// <returns></returns>
    ImagePrediction Predict(ImageData imageData);
}