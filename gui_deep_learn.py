from tkinter import Tk, Button, GROOVE
from test import IA

class Gui:
	def __init__(self, rules):
		super().__init__()

		self.root = Tk()
		self.root.title('Deep learning algorithme')
		self.root.geometry('350x150')

		self.ia = IA(
			rules['numbre of layers'], 
			rules['lenght of each layers'],
			rules['lenght of outputs'],

			cos_=True,
			rand=True,
		)

		self.btn_brut = Button(
			self.root, 
			text='Brute Force',
			command=lambda: self.ia.run_thread(lambda:self.ia.learn_by_brute_force(
				rules['rules'],
				rate_similarity=rules['rate of similarity'],
				deep=rules['deep of presition'],
			)),
			width=50,
			relief=GROOVE,
		).pack()

		self.btn_solve = Button(
			self.root, 
			text='Solving',
			command=lambda: self.ia.run_thread(lambda:self.ia.learn_by_solving(
				rules['rules'],
			)),
			width=50,
			relief=GROOVE,
		).pack()

		self.btn_experiance_once = Button(
			self.root, 
			text='Experiance Once',
			command=lambda: self.ia.run_thread(lambda:self.ia.learn_experiance_once(
				rules=rules['rules'],
				add=0.1,
			)),
			width=50,
			relief=GROOVE,
		).pack()

		self.btn_experiance_classic = Button(
			self.root, 
			text='Experiance Classic',
			command=lambda: self.ia.run_thread(lambda:self.ia.learn_by_experiance_classic(
				rules=rules['rules'],
				rate_similarity=rules['rate of similarity'],
			)),
			width=50,
			relief=GROOVE,
		).pack()

		self.btn_experiance_classic = Button(
			self.root, 
			text='Experiance 1.0 (best)',
			command=lambda: self.ia.run_thread(lambda:self.ia.learn_by_experiance_1(
				rules=rules['rules'],
			)),
			width=50,
			relief=GROOVE,
		).pack()

		self.btn_full_random = Button(
			self.root, 
			text='full_random (It might be very good)',
			command=lambda: self.ia.run_thread(lambda:self.ia.full_random(
				rules=rules['rules'],
				rate_similarity=rules['rate of similarity'],
			)),
			width=50,
			relief=GROOVE,
		).pack()

		self.root.mainloop()

if __name__ == '__main__':

	from json import load
	from sys import argv
	rules = load(open(argv[1]))

	gui = Gui(rules)

	print('\n')

	out = ''
	while out != 'sure':
		file = input('Save to (re-writing)>')
		out = input('Are you sure ? (type :sure) >')

	open('file', 'w').write('\n'.join(['\n'.join(gui.ia.layers), gui.ia.outputs]))

