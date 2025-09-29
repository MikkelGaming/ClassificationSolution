using System.IO.Enumeration;
using System.Windows;
using System.Windows.Media.Imaging;
using Microsoft.Win32;

namespace WpfImageClassifier;
public partial class MainWindow : Window
{
    private string selectedImagePath;

    public MainWindow()
    {
        InitializeComponent();
    }

    // select image button
    private void btnSelectImage_Click(object sender, RoutedEventArgs e)
    {
        OpenFileDialog ofd = new OpenFileDialog();
        ofd.Filter = "Image Files |*.jpg;*.jpeg;*.png;*.bmp";
        if (ofd.ShowDialog() == true)
        {
            selectedImagePath = ofd.FileName;
            imgPreview.Source = new BitmapImage(new Uri(selectedImagePath));
        }
        Console.WriteLine(selectedImagePath);
    }

    // start recognition
    private void btnRecognize_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(selectedImagePath))
        {
            MessageBox.Show("Please select an image");
            return;
        }
        
        MessageBox.Show($"Recognition started for: {selectedImagePath}");
    }
}