import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from src.utils import load_mnist
from src.train import train
import matplotlib
matplotlib.use('TkAgg')  # For interactive drawing

class DrawingApp:
    def __init__(self, model):
        self.model = model
        self.fig, self.ax = plt.subplots()
        self.canvas = np.zeros((28, 28))
        self.im = self.ax.imshow(self.canvas, cmap='gray', vmin=0, vmax=1)
        self.ax.set_title('Draw a digit (28x28)')
        self.fig.canvas.mpl_connect('motion_notify_event', self.draw)
        self.fig.canvas.mpl_connect('button_press_event', self.start_draw)
        self.fig.canvas.mpl_connect('button_release_event', self.stop_draw)
        self.drawing = False
        self.last_pos = None
        self.add_buttons()

    def add_buttons(self):
        ax_predict = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_clear = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.btn_predict = Button(ax_predict, 'Predict')
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_predict.on_clicked(self.predict)
        self.btn_clear.on_clicked(self.clear)

    def start_draw(self, event):
        if event.inaxes == self.ax:
            self.drawing = True
            self.last_pos = (int(event.ydata), int(event.xdata))

    def stop_draw(self, event):
        self.drawing = False
        self.last_pos = None

    def draw(self, event):
        if self.drawing and event.inaxes == self.ax and event.xdata and event.ydata:
            y, x = int(event.ydata), int(event.xdata)
            if 0 <= x < 28 and 0 <= y < 28:
                self.canvas[y, x] = 1.0
                # Optionally, draw a thicker line
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if 0 <= y+dy < 28 and 0 <= x+dx < 28:
                            self.canvas[y+dy, x+dx] = 1.0
                self.im.set_data(self.canvas)
                self.fig.canvas.draw_idle()

    def clear(self, event):
        self.canvas = np.zeros((28, 28))
        self.im.set_data(self.canvas)
        self.fig.canvas.draw_idle()

    def predict(self, event):
        # Preprocess: flatten and normalize
        x = self.canvas.reshape(1, 28*28)
        x = x / 1.0  # Already in [0,1]
        _, a1, _, a2 = self.model.forward(x)
        pred = np.argmax(a2, axis=1)[0]
        self.ax.set_title(f'Predicted: {pred}')
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    (x_train, y_train), _ = load_mnist()
    x_train = x_train / 255.0
    x_train = x_train.reshape(-1, 28*28)
    model = train(x_train, y_train)
    app = DrawingApp(model)
    plt.show()
