import os, zipfile
import sys
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import Tk
from tryconsole import Display

class simpleapp_tk(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.geometry("750x350")
        self.configure(background='#FCFCFC')
        self.initialize()

    def initialize(self):
        self.grid()
        self.Clear()

        self.resizable(True,True)
        self.textlabel=tk.Label(self, bg="#FCFCFC", text="Welcome to EDWARD!", font=("Calibri", 18, "italic"))
        self.textlabel.grid(column=1, row=1)
        self.textlabeli=tk.Label(self, bg='#FCFCFC', text=" ")
        self.textlabeli.grid(column=1, row=2)
        self.textlabel2=tk.Label(self, bg='#FCFCFC', text="This tool allows you to create customized Convolutional Neural Networks.")
        self.textlabel2.grid(column=1, row=3)
        self.textlabel3=tk.Label(self, bg='#FCFCFC', text="By adjusting basic and advanced parameters,")
        self.textlabel3.grid(column=1, row=4)
        self.textlabel4=tk.Label(self, bg='#FCFCFC', text="you can fit a huge variety of datasets.")
        self.textlabel4.grid(column=1, row=5)
        self.textlabel5=tk.Label(self, bg='#FCFCFC', text="From simple imaging to groundbreaking sequence analysis,")
        self.textlabel5.grid(column=1, row=6)
        self.textlabel6=tk.Label(self, bg='#FCFCFC', text="our tool can be used for any kind of scientific research.")
        self.textlabel6.grid(column=1, row=7)
        self.textlabel7=tk.Label(self, bg='#FCFCFC', text=" ")
        self.textlabel7.grid(column=1, row=8)
        self.dataset_arch="Dataset not provided"



        self.blankrowabove=tk.Label(self, bg="#FCFCFC", text="                                               ")
        self.blankrowabove.grid(column=0, row=0)

        self.buttoncalc=tk.Button(self, text = 'Create!', command=self.Create, bg="#774773", fg='white', activebackground="#907F9F")
        self.buttoncalc.grid(row=9, column=1, sticky="EW")

        #self.resizable(True,True)


    def Create(self):
        self.Clear()

        self.blanklabel=tk.Label(self,text='           ', bg='#FCFCFC')
        self.blanklabel.grid(column=2,row=1)

        self.label = tk.Label(self, text="Enter the amount of layers", bg='#FCFCFC')
        self.label.grid(column=3,row=1)
        self.layers = tk.StringVar(self)
        self.layers.set("2") # default value
        self.layersview = tk.OptionMenu(self, self.layers, "2", "3")
        self.layersview.grid(column=4,row=1,sticky="EW")
        self.layersview.configure(bg="white",activebackground="#A4A5AE")
        # defines the 1st entry

        self.label2 = tk.Label(self, text="Enter the batch size", bg='#FCFCFC')
        self.label2.grid(column=3,row=4)
        self.batch = tk.StringVar(self)
        self.batch.set("2") # default value
        self.batchview = tk.OptionMenu(self, self.batch, "2", "4", "8", "10", "15", "20", "30", "50", "100", "200", "500", "1000", "2000")
        self.batchview.grid(column=4,row=4,sticky="EW")
        self.batchview.configure(bg="white",activebackground="#A4A5AE")

        self.label3 = tk.Label(self, text="Enter the amount of epochs", bg='#FCFCFC')
        self.label3.grid(column=3,row=2)
        self.epochs = tk.StringVar(self)
        self.epochs.set("2") # default value
        self.epochsview = tk.OptionMenu(self, self.epochs, "10", "20", "30", "50", "100", "200", "300", "500", "1000", "2000", "5000")
        self.epochsview.grid(column=4,row=2,sticky="EW")
        self.epochsview.configure(bg="white",activebackground="#A4A5AE")

        self.label4 = tk.Label (self, text="Enter the size of pooling", bg="#FCFCFC")
        self.label4.grid(column=3,row=3,sticky="EW")
        self.pooling = tk.StringVar(self)
        self.pooling.set("2x2") # default value
        self.w = tk.OptionMenu(self, self.pooling, "2x2", "3x3", "4x4")
        self.w.grid(column=4,row=3,sticky="EW")
        self.w.configure(bg="white",activebackground="#A4A5AE")

        self.lol = tk.IntVar(False)
        self.c = tk.Checkbutton(self, text="Advanced settings", variable=self.lol, bg="#FCFCFC", activebackground="#FCFCFC", command=self.enable)
        self.c.grid(column=3,row=7)

        self.blanklabelbeforesubmit=tk.Label(self, bg='#FCFCFC',text='            ')
        self.blanklabelbeforesubmit.grid(column=3,row=11)

        self.button = tk.Button(self,text="   Submit     ", command=self.close_window, bg='#774773', fg='white', activebackground='SteelBlue1')
        self.button.grid(column=3,row=12, sticky="NE")


        self.submitdata = tk.Button(self,text="Import data", command=self.select_data, bg='#774773', fg='white', activebackground='SteelBlue1' )
        self.submitdata.grid(column=6, row=1)

        self.blankcolumn5=tk.Label(self, bg='#FCFCFC',text='            ')
        self.blankcolumn5.grid(column=5,row=1)


        self.blankcolumn5=tk.Label(self, bg='#FCFCFC',text='Please, provide data in .zip format', font = ('Calibri', 11))
        self.blankcolumn5.grid(column=6,row=2)

        self.goback=tk.Button(self, text="   Start Over     ", command=self.restart, bg='#774773', fg='white', activebackground='SteelBlue1')
        self.goback.grid(column=0,row=1, sticky="NE")


    def select_data(self):
        self.toplevel = Tk()
        self.toplevel.withdraw()
        self.filename = filedialog.askopenfilename()
        if os.path.isfile(self.filename):
            archive = zipfile.ZipFile(self.filename)
            p=os.path.abspath(self.filename)
            p=os.path.split(p)[0]
            for file in archive.namelist():
                print (file)
                archive.extract(file, p)
                if os.path.isdir(p):
                    self.dataset_arch=os.path.abspath(archive.extract(file, p))
                    print (self.dataset_arch)
                    break




    def advanced(self):
        self.gpuvar = tk.IntVar()
        self.c2 = tk.Checkbutton(self, text="I want to use GPU", variable=self.gpuvar, bg="#FCFCFC", activebackground="#FCFCFC")
        self.c2.grid(column=4,row=8,sticky="EW")

        self.filter = tk.Label(self, bg='#FCFCFC',text='Select the size of filter')
        self.filter.grid(column=3,row=9)
        self.f = tk.StringVar(self)
        self.f.set("2x2") # default value
        self.fv = tk.OptionMenu(self, self.f, "2x2", "3x3", "4x4")
        self.fv.grid(column=4,row=9,sticky="EW")
        self.fv.configure(bg="white",activebackground="#A4A5AE")

        self.learnrate = tk.Label(self, bg='#FCFCFC',text='Select the learning rate')
        self.learnrate.grid(column=3,row=10)
        self.learningrate = tk.StringVar(self)
        self.learningrate.set("0.00125") # default value
        self.learningratev = tk.OptionMenu(self, self.learningrate, "0.00125", "0.0025", "0.005", "0.01")
        self.learningratev.grid(column=4,row=10,sticky="EW")
        self.learningratev.configure(bg="white",activebackground="#A4A5AE")



    def enable(self):
        k=self.lol.get()
        print (k)
        if self.lol.get()==1:
            self.advanced()
        if self.lol.get()==0:
            try:
                self.c2.grid_forget()
            except:
                pass

    def restart(self):
        python = sys.executable
        os.execl(python, python, * sys.argv)


    def Clear(self):

        #This funcrions clears the screen. It tries to hide every component present in the program.
        #The component can be placed with grid again
        try:
            self.filter.grid_forget()
        except:
            pass
        try:
            self.learn_rate.grid_forget()
        except:
            pass
        try:
            self.submitdata.grid_forget()
        except:
            pass
        try:
            self.blankcolumn5.grid_forget()
        except:
            pass
        try:
            self.goback.grid_forget()
        except:
            pass
        try:
            self.gpuvar.grid_forget()
        except:
            pass
        try:
            self.label4.grid_forget()
        except:
            pass
        try:
            self.pooling.grid_forget()
        except:
            pass
        try:
            self.w.grid_forget()
        except:
            pass
        try:
            self.c.grid_forget()
        except:
            pass

        try:
            self.label3.grid_forget()
        except:
            pass

        try:
            self.label.grid_forget()
        except:
            pass
        try:
            self.label2.grid_forget()
        except:
            pass
        try:
            self.learningratev.grid_forget()
        except:
            pass
        try:
            self.learnrate.grid_forget()
        except:
            pass
        try:
            self.fv.grid_forget()
        except:
            pass
        try:
            self.filter.grid_forget()
        except:
            pass
        try:
            self.blanklabelbeforesubmit.grid_forget()
        except:
            pass
        try:
            self.epochsview.grid_forget()
        except:
            pass
        try:
            self.batchview.grid_forget()
        except:
            pass
        try:
            self.layersview.grid_forget()
        except:
            pass

        try:
            self.lamebutton.grid_forget()
        except:
            pass
        try:
            self.button.grid_forget()
        except:
            pass

        try:
            self.abouttext.grid_forget()
        except:
            pass
        try:
            self.entry.grid_forget()
        except:
            pass
        try:
            self.entry2.grid_forget()
        except:
            pass
        try:
            self.entry3.grid_forget()
        except:
            pass
        try:
            self.textlabel3.grid_remove()
        except:
            pass
        try:
            self.textlabel.grid_remove()
        except:
            pass
        try:
            self.textlabel2.grid_remove()
        except:
            pass
        try:
            self.textlabel4.grid_remove()
        except:
            pass
        try:
            self.textlabel5.grid_remove()
        except:
            pass
        try:
            self.textlabel6.grid_remove()
        except:
            pass
        try:
            self.textlabel7.grid_remove()
        except:
            pass

        try:
            self.buttoncalc.grid_forget()
        except:
            pass

    def close_window(self):
        self.layers = int(self.layers.get())
        self.pnlg_x=2
        self.plng_y=2
        self.filt_x=2
        self.filt_y=2
        if self.pooling.get() == '2x2':
            self.plng_x = 2
            self.plng_y = 2
        elif self.pooling.get() == '3x3':
            self.plng_x = 3
            self.plng_y = 3
        elif self.pooling.get() == '4x4':
            self.plng_x = 4
            self.plng_y = 4
        self.epochs_num=int(self.epochs.get())
        self.batch_s=int(self.batch.get())
        try:
            self.learn_rate=int(self.learningrate.get())
            if self.f.get() == '2x2':
                self.filt_x=2
                self.filt_y=2
            elif self.f.get() == '3x3':
                self.filt_x=3
                self.filt_y=3
            elif self.f.get() == '4x4':
                self.filt_x=4
                self.filt_y=4
        except:
            self.learn_rate = 0.00125
        self.msg=messagebox.showinfo("", "Your information was accepted. Press OK")
        self.destroy()
        show = Display()
        show.layers=self.layers
        show.pooling_x=self.plng_x
        show.pooling_y=self.plng_y
        show.batch=self.batch_s
        show.epochs=self.epochs_num
        show.learn_r=self.learn_rate
        show.filter_x=self.filt_x
        show.filter_y=self.filt_y
        show.dataset=self.dataset_arch
        show.mainloop()
