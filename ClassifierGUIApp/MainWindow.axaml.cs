using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using ImageClassifier.Interface_Adapter;
using ImageClassifier.Application;

namespace ClassifierGUIApp;

public partial class MainWindow : Window
{

    private string selectedImagePath;
    public MainWindow()
    {
        InitializeComponent();
    }

    // select image button
    private async void btnSelectImage_Click(object? sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            AllowMultiple = false,
            Filters =
            {
                new FileDialogFilter { Name = "Image Files", Extensions = { "jpg", "jpeg", "png", "bmp" } }
            }
        };

        var result = await dialog.ShowAsync(this);
        if (result is { Length: > 0 })
        {
            selectedImagePath = result[0];

            using (var stream = File.OpenRead(selectedImagePath))
            {
                imgPreview.Source = new Bitmap(stream);
            }

            Console.WriteLine(selectedImagePath);
        }
    }

    // start recognition
    private void btnRecognize_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(selectedImagePath))
        {
            MessageBox.Show(this, "Please select an image");
            //MessageBox.Show(MainWindow.instance, ["Congratulations! You won the game!", $"Time taken: {DateTimeOffset.UtcNow.ToUnixTimeSeconds() - startTime} Seconds"], "Victory");
            return;
        }

        ImageClassifier.Application.ImageClassifier imageClassifier = new();
        var prediction = imageClassifier.ClassifyImage(selectedImagePath);

        MessageBox.Show(this, prediction.PredictedLabel);

        //MessageBox.Show($"Recognition started for: {selectedImagePath}");
    }
}