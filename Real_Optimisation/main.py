import tkinter as tk
import time
from statistics import stdev, mean
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from CoreFunctions import *


class GeneticAlgorithm(tk.Frame):

    def algorithm(self):
        self.best_result_list = []
        self.mean_result = []
        self.deviation_result = []
        t_start = time.time()
        populations = generate_populations(int(self.entry_pop.get()))
        function_values = function(populations)
        epochs = int(self.entry_epochs.get())
        self.best_result = np.min(function_values)
        self.best_result_list.append(self.best_result)
        self.mean_result.append(mean(self.best_result_list))
        for ep in range(epochs):
            self.epoch_text.set(ep + 1)
            selected_populations = select(populations, function_values, self.variable.get(), self.variable1.get(),
                                          self.entry_selection.get())
            crossed_population = crossover(populations, selected_populations, self.entry_cross.get(),
                                           self.variable2.get())
            mutated_population = mutation(crossed_population, self.entry_mutation.get())
            function_values = function(mutated_population)

            populations = mutated_population
            if self.variable.get() == "minimisation":
                self.best_result = np.min(function_values)
            else:
                self.best_result = np.max(function_values)
            self.best_result_list.append(self.best_result)
            self.mean_result.append(mean(self.best_result_list))

            if ep > 0:
                self.deviation_result.append(stdev(self.best_result_list))

        if self.variable.get() == "minimisation":
            final_f, final_p = sort(function_values, populations, False)
        else:
            final_f, final_p = sort(function_values, populations, True)
        bestx1 = final_p[0][0]
        bestx2 = final_p[1][0]

        t_end = time.time()
        final_time = t_end - t_start
        self.time_text.set(final_time)
        self.result_text.set(final_f[0])
        self.x1_text.set(bestx1)
        self.x2_text.set(bestx2)
        self.to_csv()
        return 0

    def toPlot(self):
        if self.figures:
            self.figures[0][0].clf()
            self.figures[0][1].clf()
            self.figures[0][2].clf()

        self.t = tk.Toplevel(self)
        self.t.wm_title("Wykres 1")
        fig = plt.figure(1)
        plt.plot(self.best_result_list)

        plt.title("Wartość funkcji od iteracji", fontsize=16)
        plt.ylabel("wartości", fontsize=14)
        plt.xlabel("epoki", fontsize=14)

        canvas = FigureCanvasTkAgg(fig, master=self.t)
        canvas.get_tk_widget().pack()
        canvas.draw()

        self.t2 = tk.Toplevel(self)
        self.t2.wm_title("Wykres 2")
        fig2 = plt.figure(2)
        plt.plot(self.mean_result)

        plt.title("Średnie wartości funkcji", fontsize=16)
        plt.ylabel("wartości", fontsize=14)
        plt.xlabel("epoki", fontsize=14)

        canvas2 = FigureCanvasTkAgg(fig2, master=self.t2)
        canvas2.get_tk_widget().pack()
        canvas2.draw()

        self.t3 = tk.Toplevel(self)
        self.t3.wm_title("Wykres 3")
        fig3 = plt.figure(3)
        plt.plot(self.deviation_result)

        plt.title("Odchylenie standardowe od iteracji", fontsize=16)
        plt.ylabel("wartości", fontsize=14)
        plt.xlabel("epoki", fontsize=14)

        canvas3 = FigureCanvasTkAgg(fig3, master=self.t3)
        canvas3.get_tk_widget().pack()
        canvas3.draw()
        self.figures.append([fig, fig2, fig3])

    def to_csv(self):
        with open('res.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(map(lambda x: [x], self.best_result_list))

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.windows = []
        self.figures = []
        self.best_result_list = []
        self.mean_result = []
        self.deviation_result = []
        self.parent = parent
        self.variable = tk.StringVar(root)
        self.variable.set("minimisation")
        self.variable1 = tk.StringVar(root)
        self.variable1.set("best")
        self.variable2 = tk.StringVar(root)
        self.variable2.set("arithmetic")
        self.variable3 = tk.StringVar(root)
        self.variable3.set("uniform")
        self.epoch_text = tk.StringVar(value="")
        self.result_text = tk.StringVar(value="")
        self.x1_text = tk.StringVar(value="")
        self.x2_text = tk.StringVar(value="")
        self.time_text = tk.StringVar(value="")

        label_population = tk.Label(root, text="Wielkość populacji")
        label_epochs = tk.Label(root, text="Liczba epok")
        label_selection = tk.Label(root, text="% najlepszych / wielkość turnieju")
        label_selection_type = tk.Label(root, text="Metoda selekcji")
        label_cross = tk.Label(root, text="Prawdopodobieństwo krzyżowania")
        label_cross_type = tk.Label(root, text="Rodzaj Krzyżowania")
        label_mutation = tk.Label(root, text="Prawdopodobieństwo mutacji")
        label_mutation_type = tk.Label(root, text="Rodzaj Mutacji")
        label_elite = tk.Label(root, text="Strategia elitarna (liczba)")
        label_minmax = tk.Label(root, text="Minimalizacja/maksymalizacja:")
        label_results = tk.Label(root, text="Wyniki:", font="bold")
        label_result_epoch = tk.Label(root, text="Epoki:", font="bold", anchor="w", justify=tk.LEFT)
        self.editable_result_epoch = tk.Label(root, textvariable=self.epoch_text, font="bold", anchor=tk.E)
        label_result_best = tk.Label(root, text="Najlepszy wynik:", font="bold")
        self.editable_result_best = tk.Label(root, textvariable=self.result_text, font="bold")
        label_result_x1 = tk.Label(root, text="X1:", font="bold")
        self.editable_result_x1 = tk.Label(root, textvariable=self.x1_text, font="bold")
        label_result_x2 = tk.Label(root, text="X2:", font="bold")
        self.editable_result_x2 = tk.Label(root, textvariable=self.x2_text, font="bold")
        label_result_time = tk.Label(root, text="Czas obliczeń:", font="bold")
        self.editable_result_time = tk.Label(root, textvariable=self.time_text, font="bold")

        self.entry_pop = tk.Entry(root)
        self.entry_pop.insert(tk.END, '200')
        self.entry_epochs = tk.Entry(root)
        self.entry_epochs.insert(tk.END, '200')
        self.entry_selection = tk.Entry(root)
        self.entry_selection.insert(tk.END, '0.5')
        self.entry_selection_type = tk.OptionMenu(root, self.variable1, "best", "roulette", "tournament")
        self.entry_cross = tk.Entry(root)
        self.entry_cross.insert(tk.END, '0.8')
        self.entry_cross_type = tk.OptionMenu(root, self.variable2, "arithmetic", "heuristic")
        self.entry_mutation = tk.Entry(root)
        self.entry_mutation.insert(tk.END, '0.2')
        self.entry_mutation_type = tk.OptionMenu(root, self.variable3, "uniform")
        self.entry_elite = tk.Entry(root)
        self.entry_elite.insert(tk.END, '1')
        self.entry_minmax = tk.OptionMenu(root, self.variable, "minimisation", "maximisation")

        button = tk.Button(root, text="Oblicz", command=self.algorithm)
        button2 = tk.Button(root, text="Wyrysuj Wykres", command=self.toPlot)

        label_population.grid(row=2, column=0)
        label_epochs.grid(row=4, column=0)
        label_selection.grid(row=6, column=0)
        label_selection_type.grid(row=6, column=2)
        label_cross.grid(row=8, column=0)
        label_cross_type.grid(row=8, column=2)
        label_mutation.grid(row=10, column=0)
        label_mutation_type.grid(row=10, column=2)
        label_elite.grid(row=14, column=0)
        label_minmax.grid(row=12, column=2)
        label_results.grid(row=1, column=3, columnspan=2)
        label_result_epoch.grid(row=2, column=3, sticky=tk.W)
        self.editable_result_epoch.grid(row=2, column=4, sticky=tk.E)
        label_result_best.grid(row=3, column=3, sticky=tk.W)
        self.editable_result_best.grid(row=3, column=4, sticky=tk.E)
        label_result_x1.grid(row=4, column=3, sticky=tk.W)
        self.editable_result_x1.grid(row=4, column=4, sticky=tk.E)
        label_result_x2.grid(row=5, column=3, sticky=tk.W)
        self.editable_result_x2.grid(row=5, column=4, sticky=tk.E)
        label_result_time.grid(row=6, column=3, sticky=tk.W)
        self.editable_result_time.grid(row=6, column=4, sticky=tk.E)

        self.entry_pop.grid(row=3, column=0)
        self.entry_epochs.grid(row=5, column=0)
        self.entry_selection.grid(row=7, column=0)
        self.entry_selection_type.grid(row=7, column=2)
        self.entry_cross.grid(row=9, column=0)
        self.entry_cross_type.grid(row=9, column=2)
        self.entry_mutation.grid(row=11, column=0)
        self.entry_mutation_type.grid(row=11, column=2)
        self.entry_elite.grid(row=15, column=0)
        self.entry_minmax.grid(row=13, column=2)

        button.grid(row=18, column=1)
        button2.grid(row=18, column=3)


if __name__ == '__main__':
    root = tk.Tk()
    root.title('Projekt 2')
    root.geometry("800x420+700+350")
    app = GeneticAlgorithm(root)
    root.mainloop()
