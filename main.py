import tkinter as tk
from tkinter import ttk
from veriseti import Veriseti
from kumeleme import Kumeleme
from svm_log_naive import Svm_log_naive
from roc import Roc

class SecimEkrani:
    def __init__(self, master):
        self.master = master
        self.master.title("Yapay Zeka Projesi Seçim Ekranı")
        self.master.geometry("450x250")

        tk.Label(self.master, text="Lütfen Seçim yapınız.").pack()

        tk.Button(self.master, text="Veriseti İşlemleri (Model Eğitimi)", bg="turquoise", height="2", width="50", command=self.handle_veriseti).pack()
        tk.Button(self.master, text="Kümeleme İşlemleri (KMeans)", bg="turquoise", height="2", width="50", command=self.handle_kumeleme).pack()
        tk.Button(self.master, text="SVM,Logistic Regression,Naive Bayes", bg="turquoise", height="2", width="50", command=self.handle_svm_log_naive).pack()
        tk.Button(self.master, text="ROC Grafiği Görüntüle", bg="turquoise", height="2", width="50", command=self.handle_roc).pack()

    def handle_veriseti(self):
        Veriseti()

    def handle_kumeleme(self):
        Kumeleme()
    
    def handle_svm_log_naive(self):
        Svm_log_naive()

    def handle_roc(self):
        Roc()
    
def main():
    root = tk.Tk()
    secim_ekrani = SecimEkrani(root)
    root.mainloop()

if __name__ == "__main__":
    main()
