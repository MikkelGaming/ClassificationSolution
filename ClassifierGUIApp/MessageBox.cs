using Avalonia.Controls;

namespace ClassifierGUIApp;

public class MessageBox
{
    public static void Show(Window owner, string message, string title)
    {
        StackPanel stackPanel = new()
        {
            VerticalAlignment = Avalonia.Layout.VerticalAlignment.Center,
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center,
            Spacing = 20
        };
        stackPanel.Children.Add(new TextBlock
        {
            Text = message,
            VerticalAlignment = Avalonia.Layout.VerticalAlignment.Center,
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center
        });

        Button okButton = new()
        {
            Content = "OK",
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center
        };
        stackPanel.Children.Add(okButton);

        var dialog = new Window
        {
            Title = title,
            Content = stackPanel,
            WindowStartupLocation = WindowStartupLocation.CenterScreen,
            Height = 300,
            Width = 450
        };

        okButton.Click += (_, _) => dialog.Close();


        dialog.ShowDialog(owner);
    }

    public static void Show(Window owner, string[] message, string title)
    {
        StackPanel stackPanel = new()
        {
            VerticalAlignment = Avalonia.Layout.VerticalAlignment.Center,
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center,
            Spacing = 20
        };

        foreach (var line in message)
        {
            stackPanel.Children.Add(new TextBlock
            {
                Text = line,
                VerticalAlignment = Avalonia.Layout.VerticalAlignment.Center,
                HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center
            });
        }

        Button okButton = new()
        {
            Content = "OK",
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center
        };
        stackPanel.Children.Add(okButton);

        var dialog = new Window
        {
            Title = title,
            Content = stackPanel,
            WindowStartupLocation = WindowStartupLocation.CenterScreen,
            Height = 300,
            Width = 450
        };

        okButton.Click += (_, _) => dialog.Close();


        dialog.ShowDialog(owner);
    }
    
    public static void Show(Window owner, string message)
    {
        StackPanel stackPanel = new()
        {
            VerticalAlignment = Avalonia.Layout.VerticalAlignment.Center,
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center,
            Spacing = 20
        };
        stackPanel.Children.Add(new TextBlock
        {
            Text = message,
            VerticalAlignment = Avalonia.Layout.VerticalAlignment.Center,
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center
        });

        Button okButton = new()
        {
            Content = "OK",
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center
        };
        stackPanel.Children.Add(okButton);

        var dialog = new Window
        {
            Title = "MessageBox",
            Content = stackPanel,
            WindowStartupLocation = WindowStartupLocation.CenterScreen,
            Height = 300,
            Width = 450
        };

        okButton.Click += (_, _) => dialog.Close();


        dialog.ShowDialog(owner);
    }
}