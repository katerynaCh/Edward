import tkinter as tk
import sys
from convolutional_mlp import evaluate_lenet5
from logistic_sgd import get_image_size, get_amount_of_classes


class Display(tk.Frame):
    def __init__(self):
       tk.Frame.__init__(self)
       self.doIt = tk.Button(self,text="Start", command=self.start, background = 'black', fg='white')
       self.doIt.pack()

       self.output = tk.Text(self, width=100, height=15, background = 'black', fg='white')
       self.output.pack(side=tk.LEFT)
       sys.stdout = self

       self.scrollbar = tk.Scrollbar(self, orient="vertical", command = self.output.yview)
       self.scrollbar.pack(side=tk.RIGHT, fill="y")

       self.output['yscrollcommand'] = self.scrollbar.set

       self.count = 1
       self.configure(background='black')
       self.pack()


    def start(self):
        # testing()
        dataset = self.dataset
        print (dataset)
        image_x, image_y = get_image_size(dataset)
        amount_classes = get_amount_of_classes(dataset)

        # Pooling size
        poolsize_x = self.pooling_x
        poolsize_y = self.pooling_y
        # Learning rate
        # Epochs to be trained and batch size
        user_learning_rate = self.learn_r
        user_nepochs = self.epochs
        user_batch = self.batch

        # Size of the convolution filter windows
        user_filter_x = self.filter_x
        user_filter_y = self.filter_y

        # Treshhold for model training
        user_treshhold = 0.995

        #amount of layers given by user
        n=self.layers
        print ('Please, wait until all iterations are completed.')
        evaluate_lenet5(learning_rate=self.learn_r, n_epochs=self.epochs,
                    dataset=self.dataset,
                    nkerns=[20, 50], batch_size=self.batch)






    def write(self, txt):
        self.output.insert(tk.END,str(txt))
        self.update_idletasks()



